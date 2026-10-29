from copy import deepcopy
import unittest

from shotlab.domain import Project
from shotlab.history import History


class HistoryTests(unittest.TestCase):
    def test_split_merge_boundary_round_trip_keeps_ids_and_notes(self):
        project = Project.new(120)
        history = History()
        project.shots[0].note = "原始观察"
        original = deepcopy(project)
        for label, command in [("split", lambda: project.split(40)),
                               ("note", lambda: project.set_note(project.selected_id, "右侧")),
                               ("move", lambda: project.move_boundary(1, 41)),
                               ("merge", lambda: project.merge(0))]:
            before = deepcopy(project)
            command()
            history.record(before, project, label)
        final = deepcopy(project)
        for _ in range(4):
            project = history.restore()
            project.validate()
        self.assertEqual(project, original)
        for _ in range(4):
            project = history.restore(redo=True)
            project.validate()
        self.assertEqual(project, final)
        self.assertIn("原始观察", project.shots[0].note)
        self.assertIn("右侧", project.shots[0].note)

    def test_note_coalescing_new_branch_and_limit(self):
        project = Project.new(30)
        history = History(limit=3)
        for value in ("中", "中文", "中文输入"):
            before = deepcopy(project)
            project.set_note(project.selected_id, value)
            history.record(before, project, "note", project.selected_id)
        self.assertEqual(len(history.undo), 1)
        project = history.restore()
        self.assertEqual(project.shots[0].note, "")
        before = deepcopy(project)
        project.split(3)
        history.record(before, project, "split")
        self.assertFalse(history.redo)
        for frame in (6, 9, 12, 15):
            before = deepcopy(project)
            project.split(frame)
            history.record(before, project, "split")
        self.assertEqual(len(history.undo), 3)
