"""Standalone ONNX worker using the existing JSON-lines analysis protocol."""
import argparse
from concurrent.futures import CancelledError
import ctypes
import os
from pathlib import Path
import sys
import time

from .analysis_worker import Cancellation, emit, publish, memory_usage


def run_worker(args):
    sys.stdout.reconfigure(encoding='utf-8')
    began = time.perf_counter()
    cancelled = Cancellation(Path(args.index).parent / 'cancel')
    if sys.platform == 'win32':
        ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
    emit('loading', pid=os.getpid())
    try:
        from .detection.onnx_runtime import GPUError
    except Exception as error:
        emit('error', message=f'无法加载 ONNX 运行依赖：{error}')
        return 1
    try:
        if cancelled.is_set():
            raise CancelledError()
        from t0.media import FrameIndex
        from .detection.core import run, save_result
        from .detection.onnx import OnnxDetector
        index = FrameIndex.load(args.index)
        imported = time.perf_counter() - began
        detector = OnnxDetector(args.weights, args.device, adapter=args.adapter, device_cache=args.device_cache,
            selected=lambda device, metadata, trials: emit('device', device=device, **metadata, trials=trials))
        result = run(index, detector, start=args.start, end=args.end, cancel=cancelled,
                     progress=lambda frame, end: emit('progress', frame=frame, total=end),
                     publish=publish, cooperate=cancelled.yield_to_playback)
        if cancelled.is_set():
            raise CancelledError()
        result.timing.update(imports_seconds=imported, cancel_check_seconds=cancelled.check_seconds,
                             playback_yield_seconds=cancelled.yield_seconds)
        result.timing['other_detection_seconds'] = result.timing['detection_wall_seconds'] - sum(
            result.timing[k] for k in ('decode_seconds', 'resize_seconds', 'algorithm_seconds',
                                     'setup_seconds', 'playback_yield_seconds'))
        result.timing['worker_before_save_seconds'] = time.perf_counter() - began
        result.metadata['runtime_python'] = sys.executable
        result.metadata['runtime_prefix'] = sys.prefix
        result.metadata['python_version'] = sys.version
        result.metadata['torch_loaded'] = 'torch' in sys.modules
        result.metadata['domain_loaded'] = 'shotlab.domain' in sys.modules
        result.timing.update(memory_usage())
        saving = time.perf_counter()
        save_result(result, index, args.out)
        emit('complete', save_result_seconds=time.perf_counter()-saving,
             worker_total_seconds=time.perf_counter()-began)
        return 0
    except CancelledError:
        emit('cancelled')
        return 2
    except GPUError as error:
        emit('error', message=str(error), retry_cpu=True)
        return 3
    except Exception as error:
        emit('error', message=str(error))
        return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--weights', default=str(Path(sys.executable).parent / 'models/transnet-split.onnx'))
    parser.add_argument('--device', choices=('auto', 'cpu', 'dml'), default='auto')
    parser.add_argument('--adapter', type=int)
    parser.add_argument('--device-cache', help='Optional writable device decision JSON; never stores a Session')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int)
    return run_worker(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
