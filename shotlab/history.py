"""Bounded session history. Snapshots include identity and observations together."""

from copy import deepcopy
from dataclasses import dataclass
from time import monotonic

from .domain import Project


@dataclass
class Edit:
    before: Project
    after: Project
    label: str
    key: str | None
    time: float


class History:
    def __init__(self, limit=100):
        self.limit = limit
        self.undo = []
        self.redo = []

    def record(self, before: Project, after: Project, label: str, key: str | None = None):
        if before == after:
            return
        now = monotonic()
        if key and self.undo and self.undo[-1].key == key and now - self.undo[-1].time < 1:
            self.undo[-1].after = deepcopy(after)
            self.undo[-1].time = now
        else:
            self.undo.append(Edit(before, deepcopy(after), label, key, now))
            del self.undo[:-self.limit]
        self.redo.clear()

    def restore(self, redo=False) -> Project:
        source, target = (self.redo, self.undo) if redo else (self.undo, self.redo)
        edit = source.pop()
        target.append(edit)
        return deepcopy(edit.after if redo else edit.before)

    def extend_analysis(self, cuts, complete=False, shot_ids=None):
        for edit in self.undo + self.redo:
            for project in (edit.before, edit.after):
                project.extend_analysis(cuts, complete, shot_ids)
