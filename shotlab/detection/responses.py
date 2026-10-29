"""Shared response islands and confirmed boundary publication; no model backend."""
import numpy as np
from .core import Boundary


def response_boundaries(scores, offset, threshold=0.5, policy="midpoint"):
    """One candidate per positive response island; no minimum shot duration.

    TransNet marks a transition response on/around the last old frame.
    Convert [lo,hi) response to a partition: floor((lo+hi-1)/2) + 1.
    Raw runs and alternative peak conversion remain available for diagnostics.
    This can still merge adjacent model responses; report as a limitation.
    """
    if policy not in ("midpoint", "peak"):
        raise ValueError("Expected midpoint or peak")
    mask = np.asarray(scores) > threshold
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    result = []
    for lo, hi in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        anchor = (lo + hi - 1) // 2 if policy == "midpoint" else lo + int(np.argmax(scores[lo:hi]))
        result.append(Boundary(int(offset + anchor + 1), float(np.max(scores[lo:hi])),
                               (int(offset + lo), int(offset + hi))))
    return result


class ClosedResponses:
    """Publish only closed islands in finalized model output, never an open tail.

    The model's middle 50 outputs already have their full 100-frame context.
    Retain an island crossing a window until a later non-positive score closes
    it; an artificial chunk endpoint must not choose its midpoint prematurely.
    """

    def __init__(self, threshold, policy):
        self.threshold, self.policy = threshold, policy
        self.offset = 0
        self.pending = np.empty(0, dtype=np.float32)

    def feed(self, values, offset, final=False):
        if not len(self.pending):
            self.offset = offset
        values = np.concatenate((self.pending, values))
        safe = len(values)
        if not final and safe and values[-1] > self.threshold:
            negative = np.flatnonzero(values <= self.threshold)
            safe = int(negative[-1] + 1) if len(negative) else 0
        result = response_boundaries(values[:safe], self.offset, self.threshold, self.policy)
        self.pending = values[safe:].copy()
        self.offset += safe
        return result
