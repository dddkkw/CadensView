"""The established TransNet 100/50 source grid, independent of the backend."""
from collections import deque
from contextlib import closing

import numpy as np


def window_grid(total, start, end):
    first = max(0, start // 50 * 50 - 50)
    last = min((total + 49) // 50 * 50, (end + 49) // 50 * 50 + 50)
    return first, last


def frame_windows(source, start, end):
    total = len(source.index.pts)
    first_grid, last_grid = window_grid(total, start, end)
    lo, hi = max(0, first_grid - 25), min(total, last_grid + 25)
    buffer, head, next_frame = deque(), lo, lo
    first = last = None
    with closing(iter(source.frames(lo, hi, 48, 27))) as iterator:
        for base in range(first_grid, last_grid, 50):
            source.check_cancelled()
            while next_frame < min(total, base + 75):
                number, pixels = next(iterator)
                if first is None:
                    first = pixels
                last = pixels
                buffer.append(pixels)
                next_frame = number + 1
            window = [first if n < 0 else last if n >= total else buffer[n - head]
                      for n in range(base - 25, base + 75)]
            yield base, np.stack(window)[None]
            while head < max(0, base + 25) and buffer:
                buffer.popleft()
                head += 1
