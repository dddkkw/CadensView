"""POSIX worker priority and parent lifetime without native extensions."""
import os
import threading


def prepare_worker(parent_pid):
    if parent_pid <= 1:
        raise ValueError("A live GUI parent is required")
    try:
        os.nice(10)
    except OSError:
        pass  # Scheduling priority is optional.

    def watch():
        while os.getppid() == parent_pid:
            threading.Event().wait(.5)
        os._exit(1)  # An interrupted SQLite transaction is rolled back.

    threading.Thread(target=watch, name="parent-lifetime", daemon=True).start()
