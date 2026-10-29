"""M5/M6: deterministic, sampled SDR color analysis with no GUI dependency."""

from bisect import bisect_right
from concurrent.futures import CancelledError
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from time import perf_counter

import av
from av.video.reformatter import VideoReformatter
import numpy as np

from t0.media import FrameReader
from .cache import ColorCache
from .model import contiguous_blocks, digest
from .shots import grid


class UnsupportedColor(ValueError):
    pass


@dataclass(frozen=True)
class ColorParameters:
    sample_step_s: float = 1
    sample_width: int = 160
    palette_colors: int = 5
    lab_bins: tuple[int, int, int] = (10, 16, 16)
    palette_iterations: int = 24
    color_change_rank_count: int = 10

    def validate(self):
        if Fraction(str(self.sample_step_s)) <= 0:
            raise ValueError("Color sample interval must be positive and finite")
        if any(type(n) is not int or n < 1 for n in (self.sample_width, self.palette_colors, self.palette_iterations, self.color_change_rank_count, *self.lab_bins)):
            raise ValueError("Color sizes/counts must be positive integers")
        if len(self.lab_bins) != 3:
            raise ValueError("CIELAB histogram needs three dimensions")

    def method(self):
        return dict(asdict(self), lab_bins=list(self.lab_bins), version="sdr-srgb-lab-d65-v1", av_version=av.__version__,
                    swscale_version=list(av.library_versions['libswscale']),
                    average_color="linear RGB mean encoded as sRGB", luminance="linear RGB dot (0.2126,0.7152,0.0722)",
                    saturation="mean pixel HSV saturation in sRGB", resize="aspect-preserving AREA; rgb48le",
                    palette="weighted CIELAB k-means of occupied histogram cells; deterministic farthest-point initialization",
                    palette_positions=[.15, .5, .85], distance="sqrt Jensen-Shannon divergence, base 2; CIELAB D65 histogram",
                    pixels="include source hard subtitles and bars; exclude soft subtitles/UI")

    def feature_key(self):
        method = self.method()
        return digest({k: method[k] for k in ('version', 'av_version', 'swscale_version', 'sample_width', 'lab_bins', 'resize')})


def srgb_to_linear(rgb):
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def linear_to_srgb(rgb):
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= .0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - .055)


def linear_to_lab(rgb):
    # CIELAB relative to the same D65 reference white as sRGB/Rec.709.
    matrix = np.array([[.4123907993, .3575843394, .1804807884],
                       [.2126390059, .7151686788, .0721923154],
                       [.0193308187, .1191947798, .9505321522]])
    xyz = rgb @ matrix.T / np.array([.9504559271, 1, 1.0890577508])
    delta = 6 / 29
    value = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4 / 29)
    return np.stack((116 * value[..., 1] - 16, 500 * (value[..., 0] - value[..., 1]),
                     200 * (value[..., 1] - value[..., 2])), axis=-1)


def rgb8(linear):
    return np.rint(linear_to_srgb(np.asarray(linear)) * 255).astype(np.uint8).tolist()


def color_features(srgb, bins=(10, 16, 16), linear=None):
    """srgb is an H×W×3 float image in [0,1], not a mean color."""
    srgb = np.asarray(srgb, dtype=np.float64).reshape(-1, 3)
    linear = srgb_to_linear(srgb) if linear is None else np.asarray(linear).reshape(-1, 3)
    lab = linear_to_lab(linear)
    coordinates = np.floor((lab - [0, -128, -128]) / [100, 256, 256] * bins).astype(int)
    coordinates = np.clip(coordinates, 0, np.asarray(bins) - 1)
    flat = np.ravel_multi_index(coordinates.T, bins)
    counts = np.bincount(flat, minlength=int(np.prod(bins)))
    occupied = np.flatnonzero(counts)
    means = np.column_stack([np.bincount(flat, weights=linear[:, c], minlength=len(counts))[occupied] / counts[occupied] for c in range(3)])
    maximum, minimum = srgb.max(axis=1), srgb.min(axis=1)
    saturation = np.divide(maximum - minimum, maximum, out=np.zeros_like(maximum), where=maximum > 0)
    return dict(mean_linear_rgb=linear.mean(axis=0).tolist(), luminance=float(linear.mean(axis=0) @ [.2126, .7152, .0722]),
                saturation=float(saturation.mean()), cells=[[int(i), int(counts[i]), *mean.tolist()] for i, mean in zip(occupied, means)],
                sample_pixels=len(srgb))


def palette(features, colors=5, iterations=24):
    # Histogram cells bound clustering cost while retaining actual pixel weights.
    entries = np.asarray([cell for feature in features for cell in feature['cells']], dtype=np.float64)
    weights, linear = entries[:, 1], entries[:, 2:]
    points = linear_to_lab(linear)
    centers = [points[np.argmax(weights)]]
    for _ in range(min(colors, len(points)) - 1):
        distances = ((points[:, None, :] - centers) ** 2).sum(axis=2).min(axis=1)
        if distances.max() < 1e-12:
            break
        centers.append(points[np.argmax(distances * weights)])
    centers = np.asarray(centers)
    for _ in range(iterations):
        labels = ((points[:, None, :] - centers) ** 2).sum(axis=2).argmin(axis=1)
        updated = centers.copy()
        for i in range(len(centers)):
            mask = labels == i
            if mask.any():
                updated[i] = np.average(points[mask], axis=0, weights=weights[mask])
        if np.max(abs(updated - centers)) < 1e-5:
            centers = updated
            break
        centers = updated
    labels = ((points[:, None, :] - centers) ** 2).sum(axis=2).argmin(axis=1)
    result = []
    for i in range(len(centers)):
        mask = labels == i
        if mask.any():
            color = rgb8(np.average(linear[mask], axis=0, weights=weights[mask]))
            result.append(dict(srgb=color, hex="#" + "".join(f"{v:02x}" for v in color), pixel_share=float(weights[mask].sum() / weights.sum())))
    return sorted(result, key=lambda x: (-x['pixel_share'], x['hex']))


def histogram(feature, bins):
    values = np.zeros(int(np.prod(bins)))
    for cell, count, *_ in feature['cells']:
        values[cell] = count / feature['sample_pixels']
    return values


def color_distance(a, b, bins=(10, 16, 16)):
    left, right = histogram(a, bins), histogram(b, bins)
    mean = (left + right) / 2
    def kl(values):
        mask = values > 0
        return np.sum(values[mask] * np.log2(values[mask] / mean[mask]))
    return float(np.sqrt(np.clip((kl(left) + kl(right)) / 2, 0, 1)))


class ColorSource:
    def __init__(self, index, parameters):
        self.reader = FrameReader(index, sequential=True)
        self.reader.stream.codec_context.thread_count = 1
        self.reformatter = VideoReformatter()
        self.parameters = parameters

    def read(self, number):
        frame = self.reader.read(number)
        if frame.rotation or self.reader.stream.sample_aspect_ratio not in (None, 0, 1):
            raise UnsupportedColor("Rotation / non-square pixels are not certified for color analysis")
        if frame.color_trc in (16, 18) or frame.color_primaries == 9:
            raise UnsupportedColor("HDR / BT.2020 analysis is unsupported; no SDR tone mapping is assumed")
        if frame.colorspace not in (0, 1, 5, 6) or frame.color_range not in (1, 2):
            raise UnsupportedColor("Unknown or unsupported color matrix/range")
        if frame.color_trc not in (1, 2, 6, 13) or frame.color_primaries not in (1, 2):
            raise UnsupportedColor("Unsupported transfer function or RGB primaries")
        warnings = []
        if frame.color_primaries == 2:
            warnings.append("Unspecified primaries: assume Rec.709/sRGB D65")
        if frame.color_trc == 2:
            warnings.append("Unspecified SDR transfer: assume inverse BT.709 OETF")
        width = min(frame.width, self.parameters.sample_width)
        height = max(1, round(frame.height * width / frame.width))
        rgb = self.reformatter.reformat(frame, width=width, height=height, format="rgb48le",
                                       src_colorspace="ITU601" if frame.colorspace in (5, 6) else "ITU709",
                                       src_color_range=frame.color_range, dst_color_range=2, interpolation="AREA", threads=1).to_ndarray().astype(np.float64) / 65535
        if frame.color_trc == 13:
            linear = srgb_to_linear(rgb)
        else:
            # Explicit inverse OETF; PyAV's dst_color_trc only changes metadata.
            linear = np.where(rgb < .081, rgb / 4.5, ((rgb + .099) / 1.099) ** (1 / .45))
        feature = color_features(linear_to_srgb(linear), self.parameters.lab_bins, linear)
        feature.update(width=width, height=height, warnings=warnings, source_pts=frame.pts,
                       source_color=dict(matrix=frame.colorspace, range=frame.color_range,
                                         transfer=frame.color_trc, primaries=frame.color_primaries))
        return feature

    def close(self):
        self.reader.close()


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise CancelledError("Color analysis cancelled; project data was not changed")


def sample_frame(context, time_s):
    target = context.index.pts[0] + time_s / context.index.tb
    return max(0, min(len(context.index.pts) - 1, bisect_right(context.index.pts, target) - 1))


def analyze_frame_palette(context, number, cache_dir=None, parameters=ColorParameters(), *, force=False):
    """One exact sample's palette, independent of the multi-frame Shot palette."""
    parameters.validate()
    context.index.verify_source()
    source_key, method_key = context.source_identity['key'], parameters.feature_key()
    key = digest([source_key, method_key, number, parameters.palette_colors, parameters.palette_iterations, 'frame-palette-v1'])
    with ColorCache(cache_dir) as cache:
        result = None if force else cache.result(key)
        if result is not None:
            return dict(result, performance=dict(result_cache_hit=True, decoded_frames=0))
        feature = cache.feature(source_key, method_key, number)
        decoded = feature is None
        if decoded:
            source = ColorSource(context.index, parameters)
            try:
                feature = source.read(number)
            finally:
                source.close()
            context.index.verify_source()
            cache.remember(source_key, method_key, number, feature)
        result = dict(source_frame=number, source_pts=feature['source_pts'],
                      metadata=dict(source_identity=context.source_identity),
                      colors=palette([feature], parameters.palette_colors, parameters.palette_iterations))
        cache.publish(key, result)
        return dict(result, performance=dict(result_cache_hit=False, feature_cache_hit=not decoded, decoded_frames=int(decoded)))


def analyze_colors(context, cache_dir=None, parameters=ColorParameters(), cancel=None, progress=None, cooperate=None, include_summary=False):
    parameters.validate()
    check_cancel(cancel)
    context.index.verify_source()
    began = perf_counter()
    method = parameters.method()
    source_key = context.source_identity['key']
    method_key = parameters.feature_key()
    timeline_method = {k: v for k, v in method.items() if k not in ('palette_colors', 'palette_iterations', 'palette', 'palette_positions', 'distance', 'color_change_rank_count')}
    shots_method = {k: v for k, v in method.items() if k != 'sample_step_s'}
    timeline_key = digest([source_key, method_key, timeline_method, 'timeline'])
    shots_key = digest([source_key, method_key, shots_method, context.shot_revision, context.scope, 'shots'])
    source = None
    diagnostics = dict(decoded_samples=0, feature_cache_hits=0)
    with ColorCache(cache_dir) as cache:
        timeline, shots = cache.result(timeline_key), cache.result(shots_key)
        summary = None
        if include_summary:
            from dataclasses import asdict
            from .color_summary import SummaryParameters, summarize
            summary_parameters = SummaryParameters()
            summary_key = digest([source_key, method_key, timeline_method, asdict(summary_parameters)])
            summary = cache.result(summary_key)
        diagnostics.update(timeline_cache_hit=timeline is not None, shots_cache_hit=shots is not None)
        step = Fraction(str(parameters.sample_step_s))
        times = list(grid(context.duration, step)) if timeline is None else []
        timeline_numbers = [sample_frame(context, t) for t in times]
        shot_samples = {s.shot_id: sorted({max(s.start_frame, min(s.end_frame_exclusive - 1, sample_frame(context, s.start + s.duration * f)))
                                         for f in (Fraction(3, 20), Fraction(1, 2), Fraction(17, 20))}) for s in context.records} if shots is None else {}
        pairs = [(a, b) for block in contiguous_blocks(context.records) for a, b in zip(block, block[1:])] if shots is None else []
        summary_numbers = [s['source_frame'] for s in timeline['samples']] if include_summary and summary is None and timeline else []
        numbers = sorted(set(timeline_numbers + summary_numbers + [n for values in shot_samples.values() for n in values] +
                             [n for a, b in pairs for n in (a.end_frame_exclusive - 1, b.start_frame)]))
        features = {}
        try:
            for done, number in enumerate(numbers, 1):
                check_cancel(cancel)
                feature = cache.feature(source_key, method_key, number)
                if feature is None:
                    if cooperate:
                        cooperate()
                    if source is None:
                        source = ColorSource(context.index, parameters)
                    feature = source.read(number)
                    cache.remember(source_key, method_key, number, feature)
                    diagnostics['decoded_samples'] += 1
                else:
                    diagnostics['feature_cache_hits'] += 1
                features[number] = feature
                if progress:
                    progress(done, len(numbers))
            context.index.verify_source()
            check_cancel(cancel)
            if timeline is None:
                samples = []
                for t, n in zip(times, timeline_numbers):
                    feature = features[n]
                    samples.append(dict(video_time_s=float(t), end_time_s=float(min(context.duration, t + step)),
                                        source_frame=n, source_pts=context.index.pts[n], frame_time_s=float((context.index.pts[n] - context.index.pts[0]) * context.index.tb),
                                        srgb=rgb8(feature['mean_linear_rgb']), luminance=feature['luminance'], saturation=feature['saturation']))
                timeline = dict(metadata=context.metadata(timeline_method, whole_video=True), samples=samples,
                                movie_barcode=[dict(video_time_s=s['video_time_s'], end_time_s=s['end_time_s'], srgb=s['srgb'], source_frame=s['source_frame']) for s in samples],
                                luminance_series=[dict(video_time_s=s['video_time_s'], value=s['luminance'], source_frame=s['source_frame']) for s in samples],
                                saturation_series=[dict(video_time_s=s['video_time_s'], value=s['saturation'], source_frame=s['source_frame']) for s in samples])
            if shots is None:
                palettes = []
                for shot in context.records:
                    check_cancel(cancel)
                    if cooperate:
                        cooperate()
                    sampled = shot_samples[shot.shot_id]
                    palettes.append(dict(shot.to_dict(), sampled_frames=sampled, sample_frame_count=len(sampled),
                                         colors=palette([features[n] for n in sampled], parameters.palette_colors, parameters.palette_iterations)))
                distances = []
                for a, b in pairs:
                    check_cancel(cancel)
                    distances.append(dict(boundary_frame=b.start_frame, video_time_s=float(b.start), left_shot_id=a.shot_id,
                                  right_shot_id=b.shot_id, left_frame=a.end_frame_exclusive - 1, right_frame=b.start_frame,
                                  distance=color_distance(features[a.end_frame_exclusive - 1], features[b.start_frame], parameters.lab_bins)))
                shots = dict(metadata=context.metadata(shots_method), shot_palette=palettes, cut_color_distance=distances,
                             largest_color_changes=sorted(distances, key=lambda x: (-x['distance'], x['video_time_s']))[:parameters.color_change_rank_count])
            palette_numbers = [n for values in shot_samples.values() for n in values] + [n for a, b in pairs for n in (a.end_frame_exclusive - 1, b.start_frame)]
            for result, used, quantized in ((timeline, timeline_numbers, False), (shots, palette_numbers, True)):
                # Keep a cached group's own metadata independent of a later Shot scope.
                if 'status' not in result:
                    warnings = sorted({w for n in used for w in features[n]['warnings']})
                    status = 'empty' if quantized and not context.records else 'approximate' if warnings or quantized else 'ok'
                    result.update(status=status, computed_at=datetime.now(timezone.utc).isoformat())
                    result['metadata']['warnings'] = sorted(set(result['metadata']['warnings'] + warnings))
                    result['metadata']['sample_dimensions'] = [list(size) for size in sorted({(features[n]['width'], features[n]['height']) for n in used})]
                    result['metadata']['source_color_tags'] = [dict(values) for values in sorted({tuple(sorted(features[n]['source_color'].items())) for n in used})]
            if not diagnostics['timeline_cache_hit']:
                check_cancel(cancel)
                cache.publish(timeline_key, timeline)
            if not diagnostics['shots_cache_hit']:
                check_cancel(cancel)
                cache.publish(shots_key, shots)
            if include_summary:
                if source is not None:
                    source.close()
                    source = None
                summary, summary_diagnostics = summarize(context, timeline, features, cache, summary_key,
                                                          summary_parameters, cancel, cooperate)
                diagnostics.update(summary_diagnostics)
        except UnsupportedColor as error:
            failed = dict(status='unsupported', reason=str(error), metadata=context.metadata(shots_method))
            return dict(video_color_timeline=timeline if timeline is not None else dict(failed, metadata=context.metadata(timeline_method, whole_video=True)), shot_color_statistics=failed,
                        performance=dict(diagnostics, elapsed_s=perf_counter() - began))
        finally:
            if source is not None:
                source.close()
    result = dict(video_color_timeline=timeline, shot_color_statistics=shots,
                  performance=dict(diagnostics, elapsed_s=perf_counter() - began))
    if include_summary:
        result['video_color_summary'] = summary
    return result
