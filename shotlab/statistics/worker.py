"""Optional single-process color worker; no Qt event loop or application state."""

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import multiprocessing
import os
import time
from queue import Empty, Full


_cancel_event = None
_foreground_event = None
_progress_queue = None


def _initialize(cancel, foreground, progress, parent_pid):
    global _cancel_event, _foreground_event, _progress_queue
    _cancel_event = cancel
    _foreground_event, _progress_queue = foreground, progress
    if os.name == 'nt':
        import ctypes
        from shotlab.win_worker import exit_with_parent
        exit_with_parent(parent_pid)
        kernel = ctypes.windll.kernel32
        kernel.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)  # BELOW_NORMAL, pointer-sized pseudo handle


def _cooperate():
    if _foreground_event.is_set():
        time.sleep(.04)


def _progress(done, total):
    try:
        _progress_queue.put_nowait((done, total))
    except Full:
        pass


def _run(context, cache_dir, parameters, rebuild):
    from .rebuild import analyze_or_rebuild
    return analyze_or_rebuild(context, cache_dir, parameters, rebuild=rebuild, cancel=_cancel_event,
                              progress=_progress, cooperate=_cooperate)


class ColorWorker:
    """One active task, cooperative cancellation, and an ordinary Future result."""

    def __init__(self, cache_dir=None):
        context = multiprocessing.get_context("spawn")
        self.cancel_event = context.Event()
        self.foreground = context.Event()
        self.progress_queue = context.Queue(maxsize=1)
        self.executor = ProcessPoolExecutor(max_workers=1, mp_context=context, initializer=_initialize,
                                            initargs=(self.cancel_event, self.foreground, self.progress_queue, os.getpid()))
        self.cache_dir = cache_dir
        self.future = None

    def submit(self, context, parameters=None, rebuild=False):
        from .colors import ColorParameters
        if self.future is not None and not self.future.done():
            raise RuntimeError("A color analysis task is already running")
        self.cancel_event.clear()
        self.progress()
        self.future = self.executor.submit(_run, deepcopy(context), self.cache_dir, parameters or ColorParameters(), rebuild)
        return self.future

    def set_foreground(self, active):
        self.foreground.set() if active else self.foreground.clear()

    def progress(self):
        value = None
        try:
            while True:
                value = self.progress_queue.get_nowait()
        except Empty:
            return value

    def cancel(self):
        self.cancel_event.set()

    def close(self):
        self.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.progress_queue.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
