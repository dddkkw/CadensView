"""Isolated inference entry point. Never imports Qt or opens a project database."""

import argparse
from concurrent.futures import CancelledError
import json
import os
from pathlib import Path
import sys
import time


class Cancellation:
    def __init__(self, path):
        self.path = path
        self.next_check = 0.0
        self.cancelled = False
        self.check_seconds = 0.0
        self.yield_seconds = 0.0

    def yield_to_playback(self):
        # One bounded yield per model window. No watcher thread or busy polling.
        # Playback is foreground work; paused browsing keeps full analysis speed.
        if (self.path.parent / 'playback').exists():
            began = time.perf_counter()
            time.sleep(0.04)
            self.yield_seconds += time.perf_counter()-began

    def is_set(self):
        now = time.monotonic()
        if not self.cancelled and now >= self.next_check:
            began = time.perf_counter()
            self.cancelled = self.path.exists()
            self.check_seconds += time.perf_counter() - began
            self.next_check = now + 0.05
        return self.cancelled


def emit(kind, **values):
    print(json.dumps(dict(type=kind, **values), ensure_ascii=False), flush=True)


def publish(boundaries, complete):
    if boundaries:
        emit("partial", boundaries=[b.frame for b in boundaries])


def memory_usage():
    if sys.platform != 'win32':
        return {}
    import ctypes
    class Memory(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ('peak_working_set', 'working_set',
            'peak_paged', 'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile', 'private')]
    memory = Memory(); memory.cb = ctypes.sizeof(memory)
    if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(memory), memory.cb):
        return dict(peak_working_set_bytes=memory.peak_working_set,
                    peak_private_commit_bytes=memory.peak_pagefile)
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--device', choices=('auto', 'cpu', 'dml'), default='auto')
    parser.add_argument('--backend', choices=('onnx',), default='onnx')
    parser.add_argument('--adapter', type=int)
    parser.add_argument('--device-cache')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int)
    from .onnx_worker import run_worker
    return run_worker(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
