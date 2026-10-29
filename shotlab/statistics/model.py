"""Read-only analysis inputs and shared source-frame/time coordinates."""

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from shotlab.domain import Project
from t0.media import FrameIndex


def digest(value) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AnalysisScope:
    kind: str = "whole_video"
    shot_ids: tuple[str, ...] = ()
    time_ranges: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class StatisticsParameters:
    histogram_bin_s: float = 2
    rolling_shots: int = 15
    density_window_s: float = 60
    time_step_s: float = 1
    segment_s: float = 60
    dense_window_s: float = 30
    dense_step_s: float = 1
    dense_count: int = 3
    dense_max_overlap: float = .5
    short_threshold_s: float = 1
    short_run_min: int = 3
    rank_count: int = 5

    def validate(self):
        times = (self.histogram_bin_s, self.density_window_s, self.time_step_s, self.segment_s,
                 self.dense_window_s, self.dense_step_s, self.short_threshold_s)
        if any(Fraction(str(v)) <= 0 for v in times):
            raise ValueError("Analysis time parameters must be positive and finite")
        if any(type(v) is not int or v < 1 for v in (self.rolling_shots, self.dense_count, self.short_run_min, self.rank_count)):
            raise ValueError("Analysis count parameters must be positive integers")
        if not 0 < self.dense_max_overlap <= 1:
            raise ValueError("Overlap fraction must be in (0, 1]")


@dataclass(frozen=True)
class ShotRecord:
    shot_id: str
    shot_number: int
    start_frame: int
    end_frame_exclusive: int
    start_pts: int
    end_pts_exclusive: int
    origin_pts: int
    time_base: Fraction

    @property
    def start(self):
        return (self.start_pts - self.origin_pts) * self.time_base

    @property
    def end(self):
        return (self.end_pts_exclusive - self.origin_pts) * self.time_base

    @property
    def duration(self):
        return self.end - self.start

    @property
    def center(self):
        return (self.start + self.end) / 2

    def to_dict(self):
        return dict(shot_id=self.shot_id, shot_number=self.shot_number, start_frame=self.start_frame,
                    end_frame_exclusive=self.end_frame_exclusive, frame_count=self.end_frame_exclusive - self.start_frame,
                    start_pts=self.start_pts, end_pts_exclusive=self.end_pts_exclusive,
                    start_time_s=float(self.start), end_time_s=float(self.end), video_time_s=float(self.center),
                    duration_s=float(self.duration))


def contiguous_blocks(records):
    blocks = []
    for record in records:
        if not blocks or blocks[-1][-1].end_frame_exclusive != record.start_frame:
            blocks.append([])
        blocks[-1].append(record)
    return blocks


@dataclass
class AnalysisContext:
    index: FrameIndex
    records: tuple[ShotRecord, ...]
    source_identity: dict
    shot_revision: str
    scope: dict
    end_pts: int
    time_quality: str
    warnings: list[str]

    @property
    def duration(self):
        return (self.end_pts - self.index.pts[0]) * self.index.tb

    def metadata(self, parameters, *, whole_video=False):
        scope = self.scope
        warnings = list(self.warnings)
        if whole_video:
            scope = dict(kind="whole_video", analysis_complete=self.index.complete,
                         covered_ranges=[dict(start_frame=0, end_frame_exclusive=len(self.index.pts),
                                              start_time_s=0, end_time_s=float(self.duration))],
                         scope_duration_s=float(self.duration), pending_start_frame=None)
            warnings = [w for w in self.warnings if w.startswith("Final frame")]
            if not self.index.complete:
                warnings.append("Only an indexed prefix is available")
        return dict(schema_version=1, source_identity=self.source_identity,
                    shot_revision=None if whole_video else self.shot_revision, scope=scope,
                    time_quality=self.time_quality, parameters=parameters,
                    warnings=warnings)


def build_context(project: Project | None, index: FrameIndex, scope=AnalysisScope()) -> AnalysisContext:
    """Build facts from the committed partition, never from detector candidates or pixels."""
    if not index.pts or index.tb <= 0 or any(a >= b for a, b in zip(index.pts, index.pts[1:])):
        raise ValueError("Analysis requires a nonempty, strictly increasing display PTS index")
    warnings = []
    last_duration = index.durations[-1] if index.durations else 0
    quality = "indexed"
    if last_duration <= 0:
        if len(index.pts) == 1:
            raise ValueError("The single frame has no trustworthy display endpoint")
        # Only the last display interval is estimated, never from container/audio duration.
        differences = sorted(b - a for a, b in zip(index.pts, index.pts[1:]))
        last_duration = round(Fraction(differences[(len(differences) - 1) // 2] + differences[len(differences) // 2], 2))
        quality = "estimated"
        warnings.append("Final frame duration estimated from rounded median indexed PTS interval")
    end_pts = index.pts[-1] + last_duration
    records = []
    if project is not None:
        project.validate()
        if project.frame_count != len(index.pts):
            raise ValueError("Shot partition and source index have different frame counts")
        for row, shot in enumerate(project.shots[:project.completed_count]):
            first, end = project.interval(row)
            records.append(ShotRecord(shot.id, row + 1, first, end, index.pts[first],
                                      index.pts[end] if end < len(index.pts) else end_pts, index.pts[0], index.tb))
    revision = digest(dict(shots=[(s.shot_id, s.start_frame, s.end_frame_exclusive) for s in records],
                           pending_start=project.pending_start if project else 0))
    if scope.kind == "selected_shots":
        unknown = set(scope.shot_ids) - {s.shot_id for s in records}
        if unknown:
            raise ValueError("Selected scope contains unknown or unfinished Shots")
        records = [s for s in records if s.shot_id in scope.shot_ids]
    elif scope.kind == "time_ranges":
        ranges = [(Fraction(str(a)), Fraction(str(b))) for a, b in scope.time_ranges]
        if any(not 0 <= a < b for a, b in ranges):
            raise ValueError("Time ranges must be positive half-open source intervals")
        merged = []
        for first, end in sorted(ranges):
            if merged and first <= merged[-1][1]:
                merged[-1] = merged[-1][0], max(end, merged[-1][1])
            else:
                merged.append((first, end))
        records = [s for s in records if any(a <= s.start and s.end <= b for a, b in merged)]
        warnings.append("Time range scope includes only fully contained complete Shots; no partial Shot slicing")
    elif scope.kind != "whole_video":
        raise ValueError("Unsupported scope; OP/ED exclusion requires future explicit content regions")
    complete = bool(project and project.pending_start is None and index.complete)
    if not complete:
        warnings.append("Incomplete project/index: Shot results describe completed coverage only")
    coverage = [dict(start_frame=b[0].start_frame, end_frame_exclusive=b[-1].end_frame_exclusive,
                     start_time_s=float(b[0].start), end_time_s=float(b[-1].end)) for b in contiguous_blocks(records)]
    scope_data = dict(kind=scope.kind, shot_ids=list(scope.shot_ids), time_ranges=[list(r) for r in scope.time_ranges],
                      analysis_complete=complete, covered_ranges=coverage,
                      scope_duration_s=float(sum((s.duration for s in records), Fraction())),
                      pending_start_frame=project.pending_start if project else 0,
                      range_policy="whole_shots_fully_contained")
    identity = {k: v for k, v in asdict(index).items() if k != "elapsed"}
    source = dict(key=digest(identity), path=str(Path(index.path).resolve()), size=index.size, mtime_ns=index.mtime_ns,
                  time_base=index.time_base, origin_pts=index.pts[0], frame_count=len(index.pts), index_complete=index.complete)
    return AnalysisContext(index, tuple(records), source, revision, scope_data, end_pts, quality, warnings)
