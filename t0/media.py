"""Read-only source indexing and exact display-frame access."""

from bisect import bisect_right
from dataclasses import dataclass
from concurrent.futures import CancelledError
from fractions import Fraction
import json
from pathlib import Path
import time

import av


@dataclass
class FrameIndex:
    path: str
    size: int
    mtime_ns: int
    time_base: str
    pts: list[int]
    durations: list[int]
    keys: list[int]
    complete: bool
    metadata: dict
    elapsed: float

    @property
    def tb(self):
        return Fraction(self.time_base)

    def us(self, frame):
        return round(self.pts[frame] * self.tb * 1_000_000)

    def end_us(self, frame):
        if frame + 1 < len(self.pts):
            return self.us(frame + 1)
        return round((self.pts[frame] + self.durations[frame]) * self.tb * 1_000_000)

    def verify_source(self):
        stat = Path(self.path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (self.size, self.mtime_ns):
            raise ValueError("Source size/mtime changed; rebuild the frame index")

    def save(self, path):
        if Path(path).resolve() == Path(self.path):
            raise ValueError("Index output must not overwrite source media")
        Path(path).write_text(json.dumps(self.__dict__, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        result = cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        result.verify_source()
        return result


def build_index(path, limit=None, progress=None, cancel=None):
    path = Path(path).resolve()
    stat = path.stat()
    started = time.perf_counter()
    pts, durations, keys = [], [], []
    complete = True
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        meta = {
            "container": container.format.name, "codec": stream.codec_context.name,
            "width": stream.width, "height": stream.height,
            "pixel_format": stream.codec_context.format.name if stream.codec_context.format else "unknown",
            "average_rate": str(stream.average_rate),
            "container_start_us": container.start_time,
            "duration_us": container.duration,
            "audio_tracks": len(container.streams.audio),
            "subtitle_tracks": len(container.streams.subtitles),
        }
        for frame in container.decode(stream):
            if cancel is not None and cancel.is_set():
                raise CancelledError("Indexing cancelled")
            if limit is not None and len(pts) >= limit:
                complete = False
                break
            if frame.pts is None or (pts and frame.pts <= pts[-1]):
                raise ValueError("T0 exact indexing requires strictly increasing display PTS")
            if not pts:
                meta.update(width=frame.width, height=frame.height, pixel_format=frame.format.name)
            if frame.key_frame:
                keys.append(len(pts))
            pts.append(frame.pts)
            durations.append(frame.duration)
            if progress is not None and len(pts) % 240 == 0:
                progress(len(pts), float(frame.pts * stream.time_base))
        tb = str(stream.time_base)
    if not pts or not keys:
        raise ValueError("No decodable indexed video/keyframe")
    result = FrameIndex(str(path), stat.st_size, stat.st_mtime_ns, tb,
                        pts, durations, keys, complete, meta, time.perf_counter() - started)
    result.verify_source()
    return result


class FrameReader:
    """One owner/thread per reader; seek keyframe, then decode to exact PTS."""

    def __init__(self, index, sequential=False):
        index.verify_source()
        self.index = index
        self.container = av.open(index.path, mode="r")
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.sequential = sequential
        self.decoder = None
        self.next_frame = None

    def read(self, number):
        return self.read_range(number, number + 1)[0]

    def read_range(self, start, end):
        index = self.index
        if not 0 <= start < end <= len(index.pts):
            raise ValueError("Frame range outside the indexed source")
        reuse = (self.sequential and self.decoder is not None and self.next_frame is not None
                 and self.next_frame <= start <= self.next_frame + 24)
        if not reuse:
            key = index.keys[max(0, bisect_right(index.keys, start) - 1)]
            self.container.seek(index.pts[key], stream=self.stream, backward=True, any_frame=False)
            self.decoder = iter(self.container.decode(self.stream))
        result = []
        expected = start
        for frame in self.decoder:
            if frame.pts < index.pts[start]:
                continue
            if frame.pts != index.pts[expected]:
                raise ValueError(f"Seek/decode mismatch at frame {expected}: {frame.pts}")
            result.append(frame)
            expected += 1
            if expected == end:
                self.next_frame = end
                return result
        raise ValueError(f"Unexpected EOF while reading [{start},{end})")

    def close(self):
        self.container.close()
