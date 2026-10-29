"""Small detector contract on the existing source-frame index."""

from dataclasses import asdict, dataclass, field
from concurrent.futures import CancelledError
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Protocol

import numpy as np
from av.video.reformatter import VideoReformatter

from t0.media import FrameIndex, FrameReader


def index_digest(index: FrameIndex) -> str:
    # Exclude indexing elapsed time; include path/stat, every PTS and stream metadata.
    data = {k: v for k, v in index.__dict__.items() if k != "elapsed"}
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def file_digest(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Boundary:
    """frame is the first frame of the right Shot, never a float timestamp.

    support is a half-open model response interval, NOT an asserted dissolve.
    Scores are detector-specific, uncalibrated internal diagnostics.
    """
    frame: int
    score: float | None = None
    support: tuple[int, int] | None = None
    kind: str = "unclassified"


@dataclass
class DetectionResult:
    detector: str
    parameters: dict
    boundaries: list[Boundary]
    scores: dict[str, np.ndarray] = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class Detector(Protocol):
    def detect(self, source: "FrameSource", start: int, end: int) -> DetectionResult: ...


class FrameSource:
    """One worker owns one existing FrameReader. Only analysis resizing is new."""

    def __init__(self, index: FrameIndex, cancel=None, progress=None, publish=None, cooperate=None):
        self.index = index
        self.cancel = cancel
        self.progress = progress
        self.publish = publish
        self.cooperate = cooperate
        self.decode_seconds = 0.0
        self.resize_seconds = 0.0
        self.frames_read = 0

    def check_cancelled(self):
        if self.cancel is not None and self.cancel.is_set():
            raise CancelledError("Detection cancelled; no result committed")

    def frames(self, start, end, width, height, format="rgb24"):
        reader = FrameReader(self.index, sequential=True)
        # Keep the same bilinear pixel conversion, but reuse swscale's context.
        # VideoFrame.reformat creates a new formatter on every decoded frame.
        reformatter = VideoReformatter()
        try:
            for number in range(start, end):
                self.check_cancelled()
                began = time.perf_counter()
                frame = reader.read(number)
                self.decode_seconds += time.perf_counter() - began
                began = time.perf_counter()
                pixels = reformatter.reformat(frame, width=width, height=height, format=format,
                                              interpolation="BILINEAR").to_ndarray()
                self.resize_seconds += time.perf_counter() - began
                self.frames_read += 1
                if self.progress and self.frames_read % 1200 == 0:
                    self.progress(number, end)
                yield number, pixels
        finally:
            reader.close()


def run(index: FrameIndex, detector: Detector, start=0, end=None, cancel=None,
        progress=None, publish=None, cooperate=None) -> DetectionResult:
    if not index.complete:
        raise ValueError("Detection requires a complete source index for endpoint context")
    end = len(index.pts) if end is None else end
    if not 0 <= start < end <= len(index.pts):
        raise ValueError("Detection range must be inside the source index")
    index.verify_source()
    source = FrameSource(index, cancel, progress, publish, cooperate)
    began = time.perf_counter()
    result = detector.detect(source, start, end)
    source.check_cancelled()
    index.verify_source()
    numbers = [b.frame for b in result.boundaries]
    if any(type(n) is not int for n in numbers):
        raise ValueError("Boundary identities must be integer source frame numbers")
    if numbers != sorted(set(numbers)) or any(not start < b < end for b in numbers):
        raise ValueError("Detector returned duplicate, unsorted or out-of-range boundaries")
    for b in result.boundaries:
        if b.score is not None and not math.isfinite(b.score):
            raise ValueError("Invalid boundary score")
        if b.support is not None and not 0 <= b.support[0] < b.support[1] <= len(index.pts):
            raise ValueError("Invalid response interval")
    for name, values in result.scores.items():
        if values.shape != (end - start,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid frame-aligned scores: {name}")
    result.timing.update(detection_wall_seconds=time.perf_counter() - began,
                         decode_seconds=source.decode_seconds,
                         resize_seconds=source.resize_seconds,
                         decoded_frames=source.frames_read)
    result.metadata.update(schema=1, start=start, end=end, frame_count=len(index.pts),
                           index_sha256=index_digest(index), source_path=index.path,
                           source_size=index.size, source_mtime_ns=index.mtime_ns,
                           time_base=index.time_base, index_complete=index.complete,
                           boundary_semantics="zero-based first frame of right shot",
                           soft_subtitles="not rendered", temporal_subsampling=False)
    return result


def save_result(result: DetectionResult, index: FrameIndex, directory):
    """Exclusive result directory; never opens a project or overwrites a result."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    data = {name: getattr(result, name) for name in
            ("detector", "parameters", "timing", "metadata")}
    data["boundaries"] = [asdict(b) for b in result.boundaries]
    for item in data["boundaries"]:
        item["pts"] = index.pts[item["frame"]]
    data["score_file"] = "scores.npz"
    np.savez_compressed(directory / "scores.npz", **result.scores)
    (directory / "result.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
