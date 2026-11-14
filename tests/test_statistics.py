"""Hand-computable timing and scope contracts for M1–M4."""

from fractions import Fraction
import math
import unittest

import numpy as np

from shotlab.domain import Project, Shot
from shotlab.statistics import AnalysisScope, StatisticsParameters, build_context, compute_shot_statistics
from t0.media import FrameIndex


def example(durations, time_base="1/1", origin=100):
    pts, cursor = [], origin
    for duration in durations:
        pts.append(cursor)
        cursor += duration
    index = FrameIndex("not-opened.mkv", 0, 0, time_base, pts, durations, [0], True,
                       {"duration_us": 999999999999}, 0)
    shots = [Shot(id=f"s{i}") for i in range(len(pts))]
    project = Project(len(pts), list(range(len(pts) + 1)), shots, selected_id=shots[0].id)
    return project, index


class ShotStatisticsTests(unittest.TestCase):
    def test_overview_distribution_and_vfr_nonzero_origin(self):
        project, index = example([1, 2, 3, 4, 20])
        result = compute_shot_statistics(project, index)
        overview = result['overview']
        self.assertEqual((overview['shot_count'], overview['scope_duration_s'], overview['asl_s'], overview['msl_s']), (5, 30, 6, 3))
        self.assertAlmostEqual(overview['stddev_s'], math.sqrt(50))
        self.assertAlmostEqual(overview['cv'], math.sqrt(50) / 6)
        self.assertEqual(overview['longest_shots'][0]['shot_id'], 's4')
        hist = result['distribution']['duration_histogram']
        self.assertEqual([r['count'] for r in hist[:3]], [1, 2, 1])
        self.assertEqual(hist[-1]['start_s'], 20)
        self.assertAlmostEqual(sum(r['count_share'] for r in hist), 1)
        self.assertAlmostEqual(sum(r['duration_share'] for r in hist), 1)
        for p in (10, 25, 75, 90):
            self.assertAlmostEqual(result['distribution']['percentiles_s'][f'p{p}'], np.quantile([1, 2, 3, 4, 20], p / 100, method='linear'))
        self.assertEqual(result['shots'][0]['start_time_s'], 0)
        self.assertEqual(result['shots'][-1]['end_pts_exclusive'], 130)
        self.assertEqual(len(result['editing']['boundaries']), 4)

    def test_logarithmic_bins_count_every_shot_once_without_sub_frame_bins(self):
        lengths = [1, 2, 3, 3, 12, 24, 30, 48, 100, 2112]  # frames at 24 fps; the last is 88 s
        index = FrameIndex("not-opened.mkv", 0, 0, "1/24", list(range(sum(lengths))), [1] * sum(lengths), [0], True,
                           {"duration_us": 999999999999}, 0)
        shots = [Shot(id=f"s{i}") for i in range(len(lengths))]
        cuts = [sum(lengths[:i]) for i in range(len(lengths) + 1)]
        bins = compute_shot_statistics(Project(sum(lengths), cuts, shots, selected_id="s0"), index)['distribution']['log_duration_histogram']
        self.assertEqual(sum(b['count'] for b in bins), len(lengths))
        self.assertAlmostEqual(sum(b['count_share'] for b in bins), 1)
        self.assertAlmostEqual(sum(b['duration_share'] for b in bins), 1)
        self.assertTrue(all(a['end_s'] == b['start_s'] for a, b in zip(bins, bins[1:])))
        self.assertTrue(all(b['end_s'] - b['start_s'] >= 1 / 24 for b in bins[:-1]))
        for b in bins:
            for shot in b['shot_ids']:
                self.assertTrue(b['start_s'] <= lengths[int(shot[1:])] / 24 < b['end_s'])
        # Equal ratios, so long and short shots share one readable chart instead of 44 linear bins.
        self.assertLess(len(bins), 25)
        self.assertLessEqual(bins[0]['start_s'], 1 / 24)
        self.assertGreater(bins[-1]['end_s'], 88)

    def test_half_open_density_edges_and_midpoint_segments(self):
        p, index = example([30] * 5)
        result = compute_shot_statistics(p, index, parameters=StatisticsParameters(rolling_shots=5))
        series = result['editing']['cut_density_series']
        self.assertEqual((series[0]['cut_count'], series[0]['effective_window_seconds']), (0, 30))
        self.assertEqual(series[30]['cut_count'], 1)
        self.assertEqual(series[60]['cut_count'], 2)
        self.assertAlmostEqual(series[149]['cuts_per_minute'], 60 / 31)
        self.assertEqual([s['cut_count'] for s in result['editing']['time_segment_summary']], [1, 2, 1])
        self.assertEqual([s['shot_count'] for s in result['editing']['time_segment_summary']], [2, 2, 1])
        self.assertEqual([r['sample_count'] for r in result['editing']['rolling_asl_series']], [3, 4, 5, 4, 3])
        p, index = example([120])
        segments = compute_shot_statistics(p, index)['editing']['time_segment_summary']
        self.assertIsNone(segments[0]['asl_s'])
        self.assertEqual(segments[1]['asl_s'], 120)
        self.assertEqual(sum(s['shot_count'] for s in segments), 1)

    def test_noncontiguous_scope_and_empty_selection(self):
        p, index = example([30] * 5)
        result = compute_shot_statistics(p, index, AnalysisScope('selected_shots', ('s0', 's2', 's3')))
        self.assertEqual(result['overview']['scope_duration_s'], 90)
        self.assertEqual([c['boundary_frame'] for c in result['editing']['boundaries']], [3])
        self.assertEqual([r['sample_count'] for r in result['editing']['rolling_asl_series']], [1, 2, 2])
        empty = compute_shot_statistics(p, index, AnalysisScope('selected_shots'))
        self.assertEqual(empty['status'], 'empty')
        self.assertIsNone(empty['overview']['asl_s'])
        self.assertEqual(empty['distribution']['duration_histogram'], [])
        self.assertEqual(empty['distribution']['log_duration_histogram'], [])
        self.assertEqual(sum(b['count'] for b in result['distribution']['log_duration_histogram']), 3)
        partial = compute_shot_statistics(p, index, AnalysisScope('time_ranges', time_ranges=((10, 95),)))
        self.assertEqual([s['shot_id'] for s in partial['shots']], ['s1', 's2'])
        merged_ranges = compute_shot_statistics(p, index, AnalysisScope('time_ranges', time_ranges=((0, 40), (40, 60))))
        self.assertEqual(merged_ranges['overview']['shot_count'], 2)

    def test_partial_prefix_short_runs_and_stable_identity(self):
        p, index = example([1, 1, 1, 3, 24], '1/24')
        result = compute_shot_statistics(p, index)
        self.assertEqual(result['special']['short_shot_runs'][0]['shot_count'], 4)
        p.pending_start = p.boundaries[-2]
        partial = compute_shot_statistics(p, index)
        self.assertFalse(partial['metadata']['scope']['analysis_complete'])
        self.assertEqual(partial['metadata']['scope']['pending_start_frame'], 4)
        self.assertEqual(partial['overview']['shot_count'], 4)
        revision = partial['metadata']['shot_revision']
        p.shots[0].note = 'not a boundary change'
        self.assertEqual(build_context(p, index).shot_revision, revision)
        self.assertEqual(compute_shot_statistics(None, index)['status'], 'empty')

    def test_frame_ranges_clip_vfr_shots_and_keep_gaps_out_of_rhythm(self):
        _, index = example([1, 2, 3, 4, 5, 6, 7, 8])
        shots = [Shot(id=f's{i}') for i in range(3)]
        p = Project(8, [0, 3, 6, 8], shots, selected_id='s0')
        scope = AnalysisScope('frame_ranges', frame_ranges=((1, 2), (2, 5), (4, 5), (6, 7)))
        result = compute_shot_statistics(p, index, scope)
        self.assertEqual(result['metadata']['scope']['frame_ranges'], [[1, 5], [6, 7]])
        self.assertEqual([(s['start_frame'], s['end_frame_exclusive'], s['duration_s']) for s in result['shots']],
                         [(1, 3, 5), (3, 5, 9), (6, 7, 7)])
        overview, editing = result['overview'], result['editing']
        self.assertEqual((overview['shot_count'], overview['scope_duration_s'], overview['asl_s']), (3, 21, 7))
        self.assertEqual([c['boundary_frame'] for c in editing['boundaries']], [3])
        self.assertEqual([r['sample_count'] for r in editing['rolling_asl_series']], [2, 2, 1])
        self.assertFalse(any(15 <= r['video_time_s'] < 21 for r in editing['cut_density_series']))
        self.assertEqual(result['special']['densest_cut_windows'][0]['end_time_s'], 15)
        self.assertEqual(p.boundaries, [0, 3, 6, 8])

        fragments = compute_shot_statistics(p, index, AnalysisScope('frame_ranges', frame_ranges=((0, 1), (2, 3))))
        self.assertEqual([s['shot_id'] for s in fragments['shots']], ['s0', 's0'])
        self.assertEqual(len({s['record_key'] for s in fragments['shots']}), 2)
        self.assertEqual(fragments['overview']['asl_s'], 2)
        self.assertEqual(fragments['editing']['boundaries'], [])
        p.pending_start = 6
        self.assertEqual(compute_shot_statistics(p, index, AnalysisScope('frame_ranges', frame_ranges=((6, 8),)))['status'], 'empty')

    def test_endpoint_estimation_and_bad_index_fail_explicitly(self):
        p, index = example([2, 3, 0], '1/10')
        context = build_context(p, index)
        self.assertEqual(context.time_quality, 'estimated')
        self.assertEqual(context.duration, Fraction(7, 10))
        index.pts[1] = index.pts[0]
        with self.assertRaises(ValueError):
            build_context(p, index)

    def test_dense_windows_and_tie_breaking(self):
        p, index = example([1] * 120)
        windows = compute_shot_statistics(p, index)['special']['densest_cut_windows']
        self.assertEqual([w['start_time_s'] for w in windows], [1, 17, 33])
        self.assertEqual([w['cut_count'] for w in windows], [30, 30, 30])
        p, index = example([1100, 29950, 5000], '1/1000')
        dense = compute_shot_statistics(p, index)['special']['densest_cut_windows'][0]
        self.assertEqual((dense['start_time_s'], dense['cut_count']), (1.1, 2))


if __name__ == '__main__':
    unittest.main()
