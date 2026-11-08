"""Display geometry only; decoded source frames and PTS are never rewritten."""
from fractions import Fraction
from functools import lru_cache


def quarter_turns(rotation):
    turns = round(rotation / 90)
    if abs(rotation - turns * 90) > .01:
        raise ValueError('Non-orthogonal display rotation is unsupported')
    return turns % 4


def display_size(width, height, sar=1, rotation=0, limit=None):
    sar = Fraction(sar or 1)
    if sar <= 0:
        raise ValueError('Invalid sample aspect ratio')
    width, height = max(1, round(width * sar)), height
    if quarter_turns(rotation) % 2:
        width, height = height, width
    if limit is not None and width > limit:
        width, height = limit, max(1, round(height * limit / width))
    return width, height


def sample_size(frame, sar, limit, upscale=False):
    width, height = display_size(frame.width, frame.height, sar, frame.rotation, None if upscale else limit)
    if upscale:
        width, height = limit, max(1, round(height * limit / width))
    return (height, width) if quarter_turns(frame.rotation) % 2 else (width, height)


@lru_cache(maxsize=16)
def source_geometry(path, size, mtime_ns):
    # Legacy indices have no display metadata. Read once, without mutating the
    # index/source digest or rewriting an existing project.
    import av
    try:
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1  # Only the first frame is needed.
            frame = next(container.decode(stream))
            return dict(sample_aspect_ratio=str(stream.sample_aspect_ratio or 1), rotation=frame.rotation)
    except (av.error.FFmpegError, OSError, StopIteration):
        return {}  # Project recovery must remain available; media reports decode errors.


def index_geometry(index):
    metadata = index.metadata
    if 'sample_aspect_ratio' not in metadata:
        metadata = dict(metadata, **source_geometry(index.path, index.size, index.mtime_ns))
    return metadata


def index_display_size(index):
    metadata = index_geometry(index)
    return display_size(metadata['width'], metadata['height'], metadata.get('sample_aspect_ratio', 1), metadata.get('rotation', 0))
