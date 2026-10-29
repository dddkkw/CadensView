"""Time/area contracts for the derived whole-video groups, not new Shot rules."""

import colorsys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import av
import numpy as np

from shotlab.statistics.colors import color_features
from shotlab.statistics.color_summary import color_groups, write_png


def solid(rgb):
    return color_features(np.full((12, 16, 3), rgb, dtype=float) / 255)


class WholeVideoColorTests(unittest.TestCase):
    def test_gray_white_and_short_red_flash(self):
        result = color_groups([solid([120] * 3), solid([245] * 3), solid([255, 0, 0])], [12, 8, .1])
        self.assertEqual(len(result['groups']), 2)
        self.assertAlmostEqual(result['groups'][0]['coverage_share'], 12 / 20.1)
        self.assertGreater(result['groups'][1]['srgb'][0] - result['groups'][0]['srgb'][0], 100)
        self.assertEqual(result, color_groups([solid([120] * 3), solid([245] * 3), solid([255, 0, 0])], [12, 8, .1]))

    def test_two_large_colors_have_independent_overlapping_coverage(self):
        image = np.full((20, 20, 3), .45)
        image[:, 10:] = .95
        result = color_groups([color_features(image)], [120])
        self.assertEqual(len(result['groups']), 2)
        self.assertEqual([p['coverage_share'] for p in result['groups']], [1, 1])
        self.assertAlmostEqual(sum(p['area_time_share'] for p in result['groups']), 1)

    def test_small_saturated_corner_is_not_a_dominant_group(self):
        image = np.full((20, 20, 3), .45)
        image[:2] = [1, 0, 0]
        result = color_groups([color_features(image)], [60])
        self.assertEqual(len(result['groups']), 1)
        self.assertLess(max(result['groups'][0]['srgb']) - min(result['groups'][0]['srgb']), 2)

    def test_time_weight_over_frame_count_and_single_color(self):
        blue, gray = solid([30, 80, 190]), solid([140] * 3)
        result = color_groups([gray] * 5 + [blue], [1] * 5 + [20])
        self.assertEqual(result['groups'][0]['srgb'], [30, 80, 190])
        self.assertAlmostEqual(result['groups'][0]['coverage_share'], .8)
        self.assertEqual(len(color_groups([blue], [.2])['groups']), 1)

    def test_temporally_dispersed_colors_do_not_force_three_swatches(self):
        features = [solid(np.asarray(colorsys.hsv_to_rgb(i / 12, 1, .9)) * 255) for i in range(12)]
        self.assertEqual(color_groups(features, [1] * 12)['status'], 'dispersed')

    def test_png_keeps_actual_pixels(self):
        image = np.full((18, 32, 3), [20, 80, 190], dtype=np.uint8)
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'still.png'
            write_png(path, image)
            with av.open(str(path)) as video:
                np.testing.assert_array_equal(next(video.decode(video=0)).to_ndarray(format='rgb24'), image)


if __name__ == '__main__':
    unittest.main()
