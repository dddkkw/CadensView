"""Hold the real Windows worker handle, including when a venv launcher is used."""

import ctypes
from ctypes import wintypes
import os
import threading


def exit_with_parent(parent_pid):
    """A spawned color worker must not outlive an abruptly closed GUI."""
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x100000, False, parent_pid)  # SYNCHRONIZE, read-only lifetime wait
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    def wait():
        result = kernel.WaitForSingleObject(handle, 0xffffffff)
        kernel.CloseHandle(handle)
        if result == 0:
            os._exit(1)  # SQLite rolls back an interrupted transaction.
    threading.Thread(target=wait, name='color-parent-lifetime', daemon=True).start()


class WorkerHandle:
    def __init__(self, pid):
        if pid <= 0 or pid == os.getpid():
            raise ValueError("Invalid analysis worker PID")
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.TerminateProcess.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.OpenProcess(1, False, pid)  # PROCESS_TERMINATE only
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def terminate(self):
        if not self.kernel.TerminateProcess(self.handle, 1):
            error = ctypes.get_last_error()
            # Windows reports access denied when an opened process has already exited.
            if error != 5:
                raise ctypes.WinError(error)

    def close(self):
        self.kernel.CloseHandle(self.handle)
