"""TransNet split FP32 adapter; shares source windows and original cut rules."""
from contextlib import closing
from pathlib import Path
import time

import numpy as np

from .core import DetectionResult, file_digest
from .onnx_runtime import MODEL_SHA256, WEIGHTS_SHA256, select_session
from .responses import ClosedResponses, response_boundaries
from .windows import frame_windows, window_grid


class OnnxDetector:
    def __init__(self, model, device='auto', threads=4, adapter=None, selected=None, device_cache=None):
        self.model, self.device = Path(model), device
        self.threads, self.adapter, self.selected = threads, adapter, selected
        self.device_cache = device_cache

    def detect(self, source, start, end):
        began = time.perf_counter()
        if file_digest(self.model) != MODEL_SHA256:
            raise ValueError('ONNX model SHA256 mismatch; refusing to load')
        total = len(source.index.pts)
        grid_start, grid_end = window_grid(total, start, end)
        one, many = [], []
        streaming = ClosedResponses(.5, 'midpoint')
        session, trials, setup_seconds, algorithm_seconds = None, [], 0.0, 0.0
        with closing(frame_windows(source, start, end)) as windows:
            for base, frames in windows:
                if session is None:
                    setup = time.perf_counter()
                    session, trials = select_session(self.model, self.device, frames, self.threads,
                                                     self.adapter, source.check_cancelled, self.device_cache)
                    setup_seconds = time.perf_counter() - setup
                    if self.selected:
                        self.selected(session.device, session.metadata, trials)
                source.check_cancelled()
                compute = time.perf_counter()
                single, multiple = session.predict(frames)
                algorithm_seconds += time.perf_counter() - compute
                one.append(single[0, 25:75, 0])
                many.append(multiple[0, 25:75, 0])
                if source.publish:
                    final = base + 50 >= total
                    closed = streaming.feed(one[-1][:max(0, min(50, total - base))], base, final)
                    source.publish([b for b in closed if start < b.frame < end], final)
                if source.cooperate:
                    source.cooperate()
        single = np.concatenate(one)[:total - grid_start]
        multiple = np.concatenate(many)[:total - grid_start]
        boundaries = [b for b in response_boundaries(single, grid_start, .5, 'midpoint')
                      if start < b.frame < end]
        crop = slice(start - grid_start, end - grid_start)
        parameters = dict(threshold=.5, policy='midpoint', min_scene_len=0,
            device=session.device, threads=self.threads, input='uint8 RGB 48x27',
            interpolation='PyAV bilinear', window=100, stride=50, keep=[25, 75],
            padding='replicate at physical source endpoints only', weights_sha256=WEIGHTS_SHA256,
            model_sha256=MODEL_SHA256, precision='float32', tf32=False, backend='onnx-split')
        return DetectionResult('transnetv2', parameters, boundaries,
            {'single': single[crop], 'many': multiple[crop]},
            dict(setup_seconds=setup_seconds, algorithm_seconds=algorithm_seconds,
                 adapter_seconds= time.perf_counter() - began),
            dict(**session.metadata, context=[max(0, grid_start-25), min(total, grid_end+25)],
                 grid=[grid_start, grid_end], device_trials=trials, tf_parity_validated=False,
                 # Maintainer approved this fixed-hash model; TF parity remains unverified.
                 distribution_approved=True, weight_origin='third-party PyPI repack; see model manifest'))
