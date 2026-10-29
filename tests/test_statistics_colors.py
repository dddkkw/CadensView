"""Synthetic pixel definitions, source sampling, caching and process isolation."""

from concurrent.futures import CancelledError
from contextlib import closing
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch

import av
import numpy as np

from shotlab.domain import Project
from shotlab.statistics import AnalysisScope, analysis_bundle, build_context, compute_shot_statistics
from shotlab.statistics.colors import (ColorParameters, ColorSource, analyze_colors, analyze_frame_palette, color_distance,
                                       color_features, palette, rgb8, sample_frame, linear_to_lab)
from shotlab.statistics.worker import ColorWorker
from t0.media import build_index


def fixture(path):
    with av.open(str(path), 'w') as output:
        stream = output.add_stream('ffv1', rate=24)
        stream.width, stream.height, stream.pix_fmt = 96, 54, 'yuv444p'
        stream.codec_context.colorspace, stream.codec_context.color_range = 1, 1
        stream.codec_context.color_primaries, stream.codec_context.color_trc = 1, 13
        for number in range(48):
            array = np.zeros((54, 96, 3), np.uint8)
            array[:] = [255, 0, 0] if number < 24 else [0, 0, 255]
            frame = av.VideoFrame.from_ndarray(array, format='rgb24')
            frame.pts = number
            frame.time_base = Fraction(1, 24)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    index = build_index(path)
    project = Project.new(len(index.pts))
    for boundary in (12, 24, 36):
        project.split(boundary)
    return project, index


class ColorMathTests(unittest.TestCase):
    def test_lab_reference_and_bounded_disjoint_histograms(self):
        np.testing.assert_allclose(linear_to_lab(np.array([[0, 0, 0], [1, 1, 1]])),
                                   [[0, 0, 0], [100, 0, 0]], atol=1e-6)
        # Disjoint bins reach 1 even for nearby neutral colors. This metric
        # measures histogram overlap, not the magnitude of a Lab displacement.
        a, b = (color_features(np.full((4, 4, 3), value)) for value in (.45, .50))
        self.assertEqual(color_distance(a, b), 1)
        self.assertEqual(color_distance(a, a), 0)
        for value in np.linspace(0, 1, 21):
            score = color_distance(a, color_features(np.full((4, 4, 3), value)))
            self.assertTrue(0 <= score <= 1)
            self.assertLessEqual(float(f'{score:.3f}'), 1)

    def test_known_colors_luminance_and_pixel_saturation(self):
        for color, luminance, saturation in (([0, 0, 0], 0, 0), ([1, 1, 1], 1, 0),
                                             ([1, 0, 0], .2126, 1), ([0, 0, 1], .0722, 1)):
            feature = color_features(np.tile(color, (4, 4, 1)))
            self.assertAlmostEqual(feature['luminance'], luminance)
            self.assertAlmostEqual(feature['saturation'], saturation)
            self.assertEqual(rgb8(feature['mean_linear_rgb']), [int(x * 255) for x in color])
        red_cyan = np.array([[[1, 0, 0], [0, 1, 1]]], float)
        feature = color_features(red_cyan)
        self.assertEqual(feature['saturation'], 1)
        self.assertEqual(rgb8(feature['mean_linear_rgb']), [188, 188, 188])
        gray = color_features(np.full((2, 2, 3), .5))
        self.assertAlmostEqual(gray['luminance'], .21404114048)

    def test_palette_repeatability_and_spatially_invariant_distribution(self):
        image = np.array([[[1, 0, 0], [0, 0, 1]]], float)
        a, b = color_features(image), color_features(image[:, ::-1])
        self.assertEqual(color_distance(a, b), 0)
        colors = palette([a, b])
        self.assertEqual(colors, palette([a, b]))
        self.assertEqual({tuple(c['srgb']) for c in colors}, {(255, 0, 0), (0, 0, 255)})
        self.assertEqual(sum(c['pixel_share'] for c in colors), 1)
        self.assertEqual([c['pixel_share'] for c in colors], [.5, .5])
        red, blue = color_features(image[:, :1]), color_features(image[:, 1:])
        self.assertEqual(color_distance(red, blue), 1)
        self.assertAlmostEqual(color_distance(red, a), color_distance(a, red))
        first = color_features(np.array([[[1, 0, 0], [0, 1, 1]]], float))
        second = color_features(np.array([[[0, 1, 0], [1, 0, 1]]], float))
        self.assertEqual(first['mean_linear_rgb'], second['mean_linear_rgb'])
        self.assertGreater(color_distance(first, second), .9)


class ColorPipelineTests(unittest.TestCase):
    def test_sample_palette_is_single_frame_and_reuses_feature_and_result_cache(self):
        with TemporaryDirectory() as directory:
            project, index = fixture(Path(directory) / 'colors.mkv')
            context = build_context(project, index)
            cold = analyze_frame_palette(context, 0, directory)
            self.assertEqual(cold['performance']['decoded_frames'], 1)
            self.assertEqual(cold['source_frame'], 0)
            self.assertEqual(cold['source_pts'], index.pts[0])
            with patch('shotlab.statistics.colors.ColorSource', side_effect=AssertionError('cached sample decoded')):
                warm = analyze_frame_palette(context, 0, directory)
            self.assertEqual(cold['colors'], warm['colors'])
            self.assertTrue(warm['performance']['result_cache_hit'])
            # A second palette size derives from the existing frame feature.
            with patch('shotlab.statistics.colors.ColorSource', side_effect=AssertionError('feature hit decoded')):
                resized = analyze_frame_palette(context, 0, directory, ColorParameters(palette_colors=3))
            self.assertTrue(resized['performance']['feature_cache_hit'])
            blue = analyze_frame_palette(context, 24, directory)
            self.assertGreater(cold['colors'][0]['srgb'][0], 240)
            self.assertGreater(blue['colors'][0]['srgb'][2], 240)
            self.assertNotEqual(cold['colors'], blue['colors'])

    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        cls.project, cls.index = fixture(Path(cls.temp.name) / 'known.mkv')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_summary_reuses_features_and_warm_stills_without_decode_or_yield(self):
        context = build_context(self.project, self.index)
        with TemporaryDirectory() as directory:
            analyze_colors(context, directory)
            result = analyze_colors(context, directory, include_summary=True)
            self.assertEqual(result['performance']['decoded_samples'], 0)
            self.assertGreater(result['performance']['preview_decoded_frames'], 0)
            with patch('shotlab.statistics.color_summary.FrameReader', side_effect=AssertionError('warm still decoded')):
                warm = analyze_colors(context, directory, include_summary=True,
                                      cooperate=lambda: self.fail('warm cache yielded per sample'))
            self.assertEqual(warm['video_color_summary'], result['video_color_summary'])
            self.assertTrue(warm['performance']['summary_cache_hit'])
            self.assertEqual(warm['performance']['preview_decoded_frames'], 0)
            # A missing rebuildable image restores only that image.
            still = result['video_color_summary']['representatives'][0]
            (Path(directory) / 'previews' / still['file']).unlink()
            restored = analyze_colors(context, directory, include_summary=True)
            self.assertEqual(restored['performance']['preview_decoded_frames'], 1)
            self.assertEqual(restored['video_color_summary'], result['video_color_summary'])

    def test_pts_alignment_cache_and_boundary_invalidation(self):
        context = build_context(self.project, self.index)
        with TemporaryDirectory() as directory:
            cold = analyze_colors(context, directory)
            self.assertGreater(cold['performance']['decoded_samples'], 0)
            self.assertEqual([s['source_frame'] for s in cold['video_color_timeline']['samples']], [0, 24])
            self.assertEqual(len(cold['shot_color_statistics']['shot_palette']), 4)
            self.assertEqual(len(cold['shot_color_statistics']['cut_color_distance']), 3)
            with patch('shotlab.statistics.colors.ColorSource', side_effect=AssertionError('warm cache decoded media')):
                warm = analyze_colors(context, directory)
            self.assertEqual(warm['performance']['decoded_samples'], 0)
            self.assertEqual(warm['video_color_timeline'], cold['video_color_timeline'])
            self.assertEqual(warm['shot_color_statistics'], cold['shot_color_statistics'])
            project = deepcopy(self.project); project.move_boundary(1, 13)
            moved = analyze_colors(build_context(project, self.index), directory)
            self.assertTrue(moved['performance']['timeline_cache_hit'])
            self.assertFalse(moved['performance']['shots_cache_hit'])
            self.assertLess(moved['performance']['decoded_samples'], cold['performance']['decoded_samples'])
            self.assertEqual(moved['shot_color_statistics']['shot_palette'][0]['end_frame_exclusive'], 13)
            with self.assertRaises(ValueError):
                analysis_bundle(compute_shot_statistics(self.project, self.index), moved)
            analysis_bundle(compute_shot_statistics(project, self.index), moved)
            self.assertEqual(sample_frame(context, Fraction(1, 10)), 2)
            p, index = fixture(Path(directory) / 'one-frame-shot.mkv')
            p.split(1)
            tiny = analyze_colors(build_context(p, index), directory)
            self.assertEqual(tiny['shot_color_statistics']['shot_palette'][0]['sample_frame_count'], 1)

    def test_cancel_keeps_samples_but_does_not_publish_incomplete_results(self):
        cancel = Event()
        context = build_context(self.project, self.index)
        with TemporaryDirectory() as directory:
            with self.assertRaises(CancelledError):
                analyze_colors(context, directory, cancel=cancel,
                               progress=lambda done, total: cancel.set() if done == 2 else None)
            with closing(sqlite3.connect(Path(directory) / 'colors-v1.sqlite3')) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM features').fetchone()[0], 2)
                self.assertEqual(db.execute('SELECT count(*) FROM results').fetchone()[0], 0)
            result = analyze_colors(context, directory)
            self.assertGreaterEqual(result['performance']['feature_cache_hits'], 2)

    def test_hdr_is_explicitly_unsupported_and_movie_timeline_is_shot_independent(self):
        source = ColorSource(self.index, ColorParameters())
        frame = source.reader.read(0); frame.color_trc = 16
        try:
            with patch.object(source.reader, 'read', return_value=frame), self.assertRaisesRegex(ValueError, 'HDR'):
                source.read(0)
        finally:
            source.close()
        project = deepcopy(self.project); project.pending_start = project.boundaries[-2]
        with TemporaryDirectory() as directory:
            result = analyze_colors(build_context(project, self.index), directory)
            self.assertTrue(result['video_color_timeline']['metadata']['scope']['analysis_complete'])
            self.assertFalse(result['shot_color_statistics']['metadata']['scope']['analysis_complete'])
            self.assertEqual(len(result['shot_color_statistics']['shot_palette']), 3)

    def test_background_future_and_cooperative_cancel(self):
        with TemporaryDirectory() as directory, ColorWorker(directory) as worker:
            result = worker.submit(build_context(self.project, self.index)).result(timeout=30)
            self.assertEqual(len(result['shot_color_statistics']['shot_palette']), 4)
        with TemporaryDirectory() as directory, ColorWorker(directory) as worker:
            future = worker.submit(build_context(self.project, self.index))
            worker.cancel()
            with self.assertRaises(CancelledError):
                future.result(timeout=30)


if __name__ == '__main__':
    unittest.main()
