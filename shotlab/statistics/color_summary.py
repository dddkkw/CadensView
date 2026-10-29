"""Time-weighted color groups and six stills, derived from cached pixel distributions."""

from dataclasses import asdict, dataclass
from fractions import Fraction

import av
import numpy as np

from t0.media import FrameReader
from .colors import check_cancel, histogram, linear_to_lab, palette, rgb8, srgb_to_linear


@dataclass(frozen=True)
class SummaryParameters:
    version: str = 'time-area-groups-v1'
    clusters: int = 16
    merge_delta: float = 12
    frame_area: float = .20
    relative_area: float = .65
    minimum_coverage: float = .08
    leading_coverage: float = .18
    relative_coverage: float = .25
    preview_count: int = 6
    preview_width: int = 320


def color_groups(features, durations, parameters=SummaryParameters()):
    """Coverage is independent per group: two large colors can share the same second."""
    duration = float(sum(durations))
    weights = np.asarray(durations, dtype=float)
    bins = sorted({cell[0] for feature in features for cell in feature['cells']})
    positions = {cell: i for i, cell in enumerate(bins)}
    masses = np.zeros((len(features), len(bins)))
    linear_sum = np.zeros((len(bins), 3))
    for row, (feature, seconds) in enumerate(zip(features, weights)):
        for cell, count, *linear in feature['cells']:
            i = positions[cell]
            area = count / feature['sample_pixels']
            masses[row, i] = area
            linear_sum[i] += np.asarray(linear) * area * seconds
    totals = weights @ masses
    linear = linear_sum / totals[:, None]
    fitted = palette([dict(cells=[[i, mass, *rgb] for i, mass, rgb in zip(bins, totals, linear)])], parameters.clusters)
    centers = linear_to_lab(srgb_to_linear(np.asarray([p['srgb'] for p in fitted]) / 255))
    center_weights = np.asarray([p['pixel_share'] for p in fitted])
    # Merge close shades; brightness remains part of the distance, including neutrals.
    while len(centers) > 1:
        distance = np.linalg.norm(centers[:, None] - centers, axis=2)
        np.fill_diagonal(distance, np.inf)
        a, b = np.unravel_index(distance.argmin(), distance.shape)
        if distance[a, b] >= parameters.merge_delta:
            break
        centers[a] = np.average(centers[[a, b]], axis=0, weights=center_weights[[a, b]])
        center_weights[a] += center_weights[b]
        centers, center_weights = np.delete(centers, b, axis=0), np.delete(center_weights, b)
    labels = ((linear_to_lab(linear)[:, None] - centers) ** 2).sum(axis=2).argmin(axis=1)
    areas = np.column_stack([masses[:, labels == i].sum(axis=1) for i in range(len(centers))])
    dominant = (areas >= parameters.frame_area) & (areas >= areas.max(axis=1, keepdims=True) * parameters.relative_area)
    coverage = weights @ dominant
    leading = float(coverage.max() / duration)
    groups = []
    if leading >= parameters.leading_coverage:
        for i, seconds in enumerate(coverage):
            share = float(seconds / duration)
            if share < max(parameters.minimum_coverage, leading * parameters.relative_coverage):
                continue
            mask = labels == i
            mean = np.average(linear[mask], axis=0, weights=totals[mask])
            rgb = rgb8(mean)
            groups.append(dict(srgb=rgb, hex='#' + ''.join(f'{c:02x}' for c in rgb),
                               coverage_s=float(seconds), coverage_share=share,
                               area_time_share=float((weights @ areas[:, i]) / duration)))
    groups.sort(key=lambda p: (-p['coverage_s'], -p['area_time_share'], p['hex']))
    return dict(groups=groups[:3], duration_s=duration, status='approximate' if groups else 'dispersed',
                method=dict(asdict(parameters), denominator='sampled source duration',
                            coverage='time meeting frame_area and relative_area thresholds; independent groups may overlap',
                            weights='pixel area within frame; actual source-time interval across frames'))


def representative_candidates(samples, features, bins, parameters):
    """Rank within each real-time sixth using its pixel distribution, not its mean RGB."""
    distributions = np.asarray([histogram(f, bins) for f in features])
    seconds = np.asarray([s['end_time_s'] - s['video_time_s'] for s in samples])
    duration = samples[-1]['end_time_s']
    times = np.asarray([s['frame_time_s'] for s in samples])
    ranked = []
    for part in range(parameters.preview_count):
        first, end = duration * part / parameters.preview_count, duration * (part + 1) / parameters.preview_count
        indices = np.flatnonzero((times >= first) & (times < end))
        if not len(indices):
            continue
        target = np.average(distributions[indices], axis=0, weights=seconds[indices])
        p = distributions[indices]
        mean = (p + target) / 2
        distance = .5 * (np.sum(p * np.log2(np.maximum(p, 1e-15) / np.maximum(mean, 1e-15)), axis=1) +
                         np.sum(target * np.log2(np.maximum(target, 1e-15) / np.maximum(mean, 1e-15)), axis=1))
        candidates = []
        for i, difference in zip(indices, np.sqrt(np.maximum(0, distance))):
            luminance = features[i]['luminance']
            transition = luminance < .006 or luminance > .95
            # Single-sample flashes differ from both neighbors in an otherwise stable area.
            flash = False
            if 0 < i < len(samples) - 1:
                neighbors = np.sqrt(distributions[[i - 1, i + 1]])
                flash = np.linalg.norm(neighbors[0] - neighbors[1]) < .2 and np.linalg.norm(np.sqrt(distributions[i]) - neighbors.mean(axis=0)) > .8
            candidates.append((float(difference) + float(transition or flash), int(i)))
        candidates.sort()
        ranked.append((first, end, candidates[:3]))
    return ranked


def still_image(frame, width):
    return frame.reformat(width=width, height=max(1, round(width * frame.height / frame.width)), format='rgb24').to_ndarray()


def image_penalty(image, previous):
    # A small tie-break for dense white caption edges; no OCR or semantic score.
    rgb = image.astype(float) / 255
    gray = rgb.mean(axis=2)
    lower, upper = gray[int(len(gray) * .78):], gray[:int(len(gray) * .6)]
    white = np.all(rgb > .86, axis=2)
    caption = max(0, float(white[int(len(gray) * .78):].mean() - white[:int(len(gray) * .6)].mean()) - .03)
    caption *= max(0, float(np.abs(np.diff(lower, axis=1)).mean() - np.abs(np.diff(upper, axis=1)).mean()))
    duplicate = previous is not None and np.abs(image.astype(float) - previous.astype(float)).mean() < 5
    return min(.12, caption * 8) + (.2 if duplicate else 0)


def write_png(path, image):
    frame = av.VideoFrame.from_ndarray(image, format='rgb24')
    codec = av.CodecContext.create('png', 'w')
    codec.width, codec.height, codec.pix_fmt = frame.width, frame.height, 'rgb24'
    codec.time_base = Fraction(1, 1)
    codec.thread_count = 1
    frame.pts = 0
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(b''.join(bytes(p) for p in [*codec.encode(frame), *codec.encode(None)]))
    temporary.replace(path)


def summarize(context, timeline, features, cache, key, parameters=SummaryParameters(), cancel=None, cooperate=None):
    existing = cache.result(key)
    folder = cache.directory / 'previews'
    folder.mkdir(exist_ok=True)
    diagnostics = dict(summary_cache_hit=existing is not None, preview_decoded_frames=0)
    if existing and all((folder / s['file']).is_file() for s in existing['representatives']):
        return existing, diagnostics
    samples = timeline['samples']
    reader = None
    try:
        if existing:
            chosen = existing['representatives']
            for still in chosen:
                check_cancel(cancel)
                if cooperate:
                    cooperate()
                if not (folder / still['file']).is_file():
                    if reader is None:
                        reader = FrameReader(context.index)
                        reader.stream.codec_context.thread_count = 1
                    image = still_image(reader.read(still['source_frame']), parameters.preview_width)
                    diagnostics['preview_decoded_frames'] += 1
                    write_png(folder / still['file'], image)
            return existing, diagnostics
        values = [features[s['source_frame']] for s in samples]
        summary = color_groups(values, [s['end_time_s'] - s['video_time_s'] for s in samples], parameters)
        summary['representatives'] = []
        previous = None
        for first, end, candidates in representative_candidates(samples, values, timeline['metadata']['parameters']['lab_bins'], parameters):
            choices = []
            for difference, i in candidates:
                check_cancel(cancel)
                if cooperate:
                    cooperate()
                if reader is None:
                    reader = FrameReader(context.index)
                    reader.stream.codec_context.thread_count = 1
                image = still_image(reader.read(samples[i]['source_frame']), parameters.preview_width)
                diagnostics['preview_decoded_frames'] += 1
                choices.append((difference + image_penalty(image, previous), i, image))
            _, i, previous = min(choices, key=lambda choice: (choice[0], choice[1]))
            sample = samples[i]
            filename = f'{key[:20]}-{sample["source_frame"]}.png'
            write_png(folder / filename, previous)
            summary['representatives'].append(dict(source_frame=sample['source_frame'], video_time_s=sample['frame_time_s'],
                                                   start_time_s=first, end_time_s=end, file=filename))
        summary['metadata'] = timeline['metadata']
        check_cancel(cancel)
        cache.publish(key, summary)
        return summary, diagnostics
    finally:
        if reader is not None:
            reader.close()
