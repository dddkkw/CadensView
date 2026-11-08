"""Read-only source indexing and exact display-frame access."""

from bisect import bisect_right
from dataclasses import dataclass
from concurrent.futures import CancelledError
from fractions import Fraction
import json
from math import lcm
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import time

import av


def configure_video_decoder(stream):
    stream.thread_type = "AUTO"
    stream.codec_context.thread_count = min(8, max(1, os.cpu_count() or 4))


def atomic_write_json(path, data, **options):
    path = Path(path)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, **options)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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

    @property
    def timestamp_quality(self):
        return "recovered" if "timestamp_recovery" in self.metadata else "exact"

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
        path = Path(path).resolve()
        if path == Path(self.path).resolve():
            raise ValueError("Index output must not overwrite source media")
        atomic_write_json(path, self.__dict__)

    @classmethod
    def load(cls, path):
        try:
            result = cls(**json.loads(Path(path).read_text(encoding="utf-8")))
            if (not isinstance(result.path, str) or not result.path
                    or type(result.size) is not int or type(result.mtime_ns) is not int
                    or type(result.complete) is not bool or not isinstance(result.metadata, dict)
                    or not isinstance(result.time_base, str) or result.tb <= 0):
                raise ValueError("invalid source or time-base fields")
            for values in (result.pts, result.durations, result.keys):
                if not isinstance(values, list) or not values or any(type(n) is not int for n in values):
                    raise ValueError("expected nonempty integer PTS, duration and keyframe arrays")
            if (len(result.pts) != len(result.durations)
                    or any(a >= b for a, b in zip(result.pts, result.pts[1:]))
                    or any(a >= b for a, b in zip(result.keys, result.keys[1:]))
                    or not 0 <= result.keys[0] <= result.keys[-1] < len(result.pts)):
                raise ValueError("inconsistent frame or keyframe sequence")
        except (ValueError, TypeError, ZeroDivisionError) as error:
            raise ValueError(f"Invalid frame index {path}: {error}. Rebuild it from the source video.") from error
        result.verify_source()
        return result


class IndexErrorDetail(ValueError):
    def __init__(self, code, details=None):
        super().__init__(code)
        self.code = code
        self.details = details or {}


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise CancelledError("Indexing cancelled")


def scan_timestamps(path, flags, limit, progress, cancel, meta):
    records, keys = [], []
    complete = True
    check_cancel(cancel)
    with av.open(str(path), mode="r", options={"fflags": flags}) as container:
        if not container.streams.video:
            raise IndexErrorDetail("no_video")
        stream = container.streams.video[0]
        if stream.codec_context is None:
            raise IndexErrorDetail("unsupported")
        configure_video_decoder(stream)
        stream_base = stream.time_base
        meta.update({
            "container": container.format.name, "codec": stream.codec_context.name,
            "width": stream.width, "height": stream.height,
            "pixel_format": stream.codec_context.format.name if stream.codec_context.format else "unknown",
            "average_rate": str(stream.average_rate),
            "container_start_us": container.start_time,
            "duration_us": container.duration,
            "audio_tracks": len(container.streams.audio),
            "subtitle_tracks": len(container.streams.subtitles),
        })
        for frame in container.decode(stream):
            check_cancel(cancel)
            if limit is not None and len(records) >= limit:
                complete = False
                break
            base = frame.time_base or stream_base
            if not base or base <= 0:
                raise IndexErrorDetail("time_base", dict(first_frame=len(records), original_pts=frame.pts,
                                                         time_base=str(base), count=1, types=["time_base"]))
            if not records:
                meta.update(width=frame.width, height=frame.height, pixel_format=frame.format.name,
                            sample_aspect_ratio=str(stream.sample_aspect_ratio or 1), rotation=frame.rotation)
            if frame.key_frame:
                keys.append(len(records))
            records.append((frame.pts, frame.duration or 0, base, frame.dts))
            if progress is not None and len(records) % 240 == 0:
                progress(len(records), -1.0 if frame.pts is None else float(frame.pts * base))
    check_cancel(cancel)
    if not records or not keys:
        raise IndexErrorDetail("no_frames")
    return records, keys, complete, stream_base if stream_base and stream_base > 0 else records[0][2]


def normalize_timestamps(records, stream_base):
    ratios = {base: base / stream_base for _, _, base, _ in records}
    denominator = lcm(*(ratio.denominator for ratio in ratios.values()))
    factors = {base: int(ratio * denominator) for base, ratio in ratios.items()}
    return (stream_base / denominator,
            [None if pts is None else pts * factors[base] for pts, _, base, _ in records],
            [duration * factors[base] for _, duration, base, _ in records])


def timestamp_anomalies(pts):
    anomalies = {}
    previous = None
    for i, current in enumerate(pts):
        if current is None:
            anomalies[i] = "missing_pts"
        elif previous is not None and current <= previous:
            anomalies[i] = "duplicate_pts" if current == previous else "backward_pts"
        if current is not None:
            previous = current
    return anomalies


def fill_missing(pts, durations, missing):
    """Only fill interior gaps whose decoded durations exactly meet both anchors."""
    if len(missing) > 32:
        return None
    restored = pts.copy()
    run = 0
    for i in missing:
        run = run + 1 if i > 0 and pts[i - 1] is None else 1
        if i == 0 or run > 4 or durations[i - 1] <= 0 or durations[i] <= 0:
            return None
        restored[i] = restored[i - 1] + durations[i - 1]
        if i + 1 == len(pts) or (pts[i + 1] is not None and restored[i] + durations[i] != pts[i + 1]):
            return None
    return restored


def build_index(path, limit=None, progress=None, cancel=None):
    path = Path(path).resolve()
    started = time.perf_counter()
    meta = {}
    try:
        check_cancel(cancel)
        stat = path.stat()
        records, keys, complete, stream_base = scan_timestamps(path, "-genpts", limit, progress, cancel, meta)
        base, pts, durations = normalize_timestamps(records, stream_base)
        anomalies = timestamp_anomalies(pts)
        if anomalies:
            first = next(iter(anomalies))
            raw, _, raw_base, _ = records[first]
            details = dict(first_frame=first, count=len(anomalies), original_pts=raw,
                           time_base=str(raw_base), types=sorted(set(anomalies.values())))
            restored, method = None, None
            if set(anomalies.values()) == {"missing_pts"}:
                # A second pass can recover presentation order from demuxer evidence.
                # Frame/packet DTS themselves are never substituted for PTS.
                if any(dts is not None for _, _, _, dts in records):
                    generated, new_keys, new_complete, new_base = scan_timestamps(
                        path, "+genpts", limit, None, cancel, {})
                    gen_base, gen_pts, gen_durations = normalize_timestamps(generated, new_base)
                    if (new_keys == keys and new_complete == complete and len(gen_pts) == len(pts)
                            and not timestamp_anomalies(gen_pts)
                            and all(a is None or a * base == b * gen_base for a, b in zip(pts, gen_pts))):
                        base, durations, restored, method = gen_base, gen_durations, gen_pts, "genpts"
                if restored is None:
                    restored, method = fill_missing(pts, durations, anomalies), "duration_anchors"
            if restored is None:
                raise IndexErrorDetail(anomalies[first], details)
            meta["timestamp_recovery"] = dict(details, method=method, missing=list(anomalies))
            pts = restored
        other_bases = [[i, pts_value, str(frame_base)] for i, (pts_value, _, frame_base, _) in enumerate(records)
                       if frame_base != base]
        if other_bases:
            meta["original_pts_overrides"] = other_bases
        result = FrameIndex(str(path), stat.st_size, stat.st_mtime_ns, str(base),
                        pts, durations, keys, complete, meta, time.perf_counter() - started)
        try:
            result.verify_source()
        except ValueError as error:
            raise IndexErrorDetail("source_changed") from error
        check_cancel(cancel)
    except (av.error.DecoderNotFoundError, av.error.DemuxerNotFoundError, av.error.PatchWelcomeError) as error:
        raise IndexErrorDetail("unsupported", dict(meta, error_type=type(error).__name__)) from error
    except OSError as error:
        raise IndexErrorDetail("io", dict(meta, error_type=type(error).__name__, errno=error.errno)) from error
    except av.FFmpegError as error:
        raise IndexErrorDetail("decode", dict(meta, error_type=type(error).__name__, errno=error.errno)) from error
    except IndexErrorDetail as error:
        error.details = dict(meta, **error.details)
        raise
    return result


class FrameReader:
    """Seek exact timestamps, or replay a restored source by display ordinal."""

    def __init__(self, index, sequential=False):
        index.verify_source()
        self.index = index
        self.sequential = sequential
        recovery = index.metadata.get("timestamp_recovery", {})
        self.ordinal = recovery.get("method") in ("duration_anchors", "genpts")
        self.genpts = recovery.get("method") != "duration_anchors"
        self.missing = set(recovery.get("missing", ())) if not self.genpts else set()
        self.open()

    def open(self):
        self.container = av.open(self.index.path, mode="r", options={"fflags": "+genpts" if self.genpts else "-genpts"})
        self.stream = self.container.streams.video[0]
        configure_video_decoder(self.stream)
        self.decoder = None
        self.next_frame = None

    def read(self, number):
        return self.read_range(number, number + 1)[0]

    def read_range(self, start, end):
        index = self.index
        if not 0 <= start < end <= len(index.pts):
            raise ValueError("Frame range outside the indexed source")
        if self.ordinal:
            return self.read_by_ordinal(start, end)
        reuse = (self.sequential and self.decoder is not None and self.next_frame is not None
                 and self.next_frame <= start <= self.next_frame + 24)
        base = index.tb
        if not reuse:
            key = index.keys[max(0, bisect_right(index.keys, start) - 1)]
            offset = index.pts[key] if base == self.stream.time_base else (
                index.pts[key] * base // self.stream.time_base)
            self.container.seek(offset, stream=self.stream, backward=True, any_frame=False)
            self.decoder = iter(self.container.decode(self.stream))
        result = []
        expected = start
        for frame in self.decoder:
            frame_base = frame.time_base or self.stream.time_base
            if frame.pts is None:
                raise ValueError(f"Missing decoded PTS while seeking source frame {expected}")
            tick = frame.pts if frame_base == base else frame.pts * frame_base / base
            if tick < index.pts[start]:
                continue
            if tick != index.pts[expected]:
                raise ValueError(f"Seek/decode mismatch at frame {expected}: {tick}")
            result.append(frame)
            expected += 1
            if expected == end:
                self.next_frame = end
                return result
        raise ValueError(f"Unexpected EOF while reading [{start},{end})")

    def read_by_ordinal(self, start, end):
        # Restored streams may emit different PTS after a seek. Replay from the
        # beginning for backwards jumps, keeping forward decoding single-pass.
        if self.decoder is None or start < self.next_frame:
            if self.decoder is not None:
                threads = self.stream.codec_context.thread_count
                self.close()
                self.open()
                self.stream.codec_context.thread_count = threads
            self.decoder = iter(self.container.decode(self.stream))
            self.next_frame = 0
        result, base = [], self.index.tb
        while self.next_frame < end:
            i = self.next_frame
            frame = next(self.decoder, None)
            if frame is None:
                self.next_frame = len(self.index.pts)
                raise ValueError(f"Unexpected EOF at source frame {i}")
            expected = None if i in self.missing else self.index.pts[i] * base
            actual = None if frame.pts is None else frame.pts * (frame.time_base or self.stream.time_base)
            if expected != actual:
                self.next_frame = len(self.index.pts)
                raise ValueError(f"Decode order differs from the restored index at source frame {i}")
            if i >= start:
                result.append(frame)
            self.next_frame += 1
        return result

    def close(self):
        self.container.close()
