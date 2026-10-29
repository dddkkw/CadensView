"""Project editing rules; no Qt, decoder or storage dependency."""

from bisect import bisect_right
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class Shot:
    id: str = field(default_factory=lambda: uuid4().hex)
    note: str = ""


@dataclass
class Project:
    frame_count: int
    boundaries: list[int]
    shots: list[Shot]
    id: str = field(default_factory=lambda: uuid4().hex)
    current_frame: int = 0
    selected_id: str = ""
    # A pending tail reserves coverage, but is not a detected/editable Shot.
    # Completed projects retain the original full partition contract.
    pending_start: int | None = None

    @property
    def completed_count(self):
        return len(self.shots) - (self.pending_start is not None)

    def extend_analysis(self, cuts, complete=False, shot_ids=None):
        """Append finalized intervals without changing selection, IDs or notes.

        Only the reserved, uneditable tail may be split by automatic analysis.
        The same operation can rebase session history snapshots safely.
        """
        if self.pending_start is None:
            raise ValueError("分析已完成，不能覆盖已有镜头")
        if (cuts != sorted(set(cuts)) or
                any(type(c) is not int or not self.pending_start < c < self.frame_count for c in cuts)):
            raise ValueError("增量边界必须位于待分析尾段内")
        additions = [Shot(id=identity) for identity in shot_ids] if shot_ids is not None else [Shot() for _ in cuts]
        if len(additions) != len(cuts):
            raise ValueError("增量镜头 ID 数量不符")
        for cut, shot in zip(cuts, additions):
            self.boundaries.insert(-1, cut)
            self.shots.append(shot)
            self.pending_start = cut
        if complete:
            self.pending_start = None
        self.validate()
        return [shot.id for shot in additions]

    @classmethod
    def new(cls, frame_count: int):
        shot = Shot()
        return cls(frame_count, [0, frame_count], [shot], selected_id=shot.id)

    def validate(self):
        if (self.frame_count < 1 or len(self.boundaries) != len(self.shots) + 1
                or self.boundaries[0] != 0 or self.boundaries[-1] != self.frame_count
                or any(a >= b for a, b in zip(self.boundaries, self.boundaries[1:]))):
            raise ValueError("Shot boundaries must partition every source frame exactly once")
        ids = [shot.id for shot in self.shots]
        if len(set(ids)) != len(ids) or self.selected_id not in ids:
            raise ValueError("Invalid Shot identity or selection")
        if not 0 <= self.current_frame < self.frame_count:
            raise ValueError("Playback position outside source")
        if self.pending_start is not None and (self.pending_start != self.boundaries[-2] or self.shots[-1].note):
            raise ValueError("待分析尾段必须独立且不能有镜头笔记")

    def shot_at(self, frame: int) -> int:
        if not 0 <= frame < self.frame_count:
            raise ValueError("Frame outside source")
        return bisect_right(self.boundaries, frame) - 1

    def row(self, shot_id: str) -> int:
        return next(i for i, shot in enumerate(self.shots) if shot.id == shot_id)

    def interval(self, row: int) -> tuple[int, int]:
        return self.boundaries[row], self.boundaries[row + 1]

    def split(self, frame: int) -> int:
        row = self.shot_at(frame)
        if row >= self.completed_count:
            raise ValueError("请等待此区域分析完成")
        if frame == self.boundaries[row]:
            raise ValueError("A boundary already exists at this frame")
        # Existing observations remain on the stable left Shot. The new Shot
        # starts empty; copying a whole-Shot note would invent its applicability.
        self.boundaries.insert(row + 1, frame)
        self.shots.insert(row + 1, Shot())
        self.selected_id = self.shots[row + 1].id
        self.current_frame = frame
        return row + 1

    def merge(self, left: int) -> int:
        if not 0 <= left < self.completed_count - 1:
            raise ValueError("No adjacent Shot to merge")
        first, second = self.shots[left:left + 2]
        if second.note:
            start, end = self.interval(left + 1)
            separator = f"合并前范围 [{start}, {end}) · {second.id[:8]}"
            first.note = "\n\n".join(filter(None, [first.note, separator, second.note]))
        del self.boundaries[left + 1]
        del self.shots[left + 1]
        self.selected_id = first.id
        return left

    def move_boundary(self, boundary: int, frame: int):
        if not 0 < boundary < self.completed_count:
            raise ValueError("Source endpoints cannot be moved")
        if not self.boundaries[boundary - 1] < frame < self.boundaries[boundary + 1]:
            raise ValueError("Both neighbouring Shots must retain at least one frame")
        self.boundaries[boundary] = frame

    def set_note(self, shot_id: str, text: str):
        row = self.row(shot_id)
        if row >= self.completed_count:
            raise ValueError("待分析区域不能添加镜头笔记")
        self.shots[row].note = text
