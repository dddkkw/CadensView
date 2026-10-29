import json
from dataclasses import replace
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from shotlab.domain import Project
from shotlab.storage import ProjectStore, SaveConflict
from t0.media import FrameIndex


class ProjectTests(unittest.TestCase):
    def test_shared_boundaries_and_stable_notes(self):
        p = Project.new(120)
        first = p.shots[0].id
        p.set_note(first, "左侧观察")
        p.split(50)
        second = p.shots[1].id
        self.assertEqual(p.shots[1].note, "")
        p.set_note(second, "右侧观察")
        p.move_boundary(1, 63)
        self.assertEqual([p.interval(0), p.interval(1)], [(0, 63), (63, 120)])
        self.assertEqual([(s.id, s.note) for s in p.shots], [(first, "左侧观察"), (second, "右侧观察")])
        p.merge(0)
        self.assertEqual(p.shots[0].id, first)
        self.assertIn("左侧观察", p.shots[0].note)
        self.assertIn("右侧观察", p.shots[0].note)
        self.assertIn("[63, 120)", p.shots[0].note)
        p.validate()

    def test_invalid_edits_leave_partition_unchanged(self):
        p = Project.new(10)
        p.split(1)
        before = list(p.boundaries), [(s.id, s.note) for s in p.shots]
        for boundary, frame in [(0, 4), (2, 5), (1, 0), (1, 10)]:
            with self.assertRaises(ValueError):
                p.move_boundary(boundary, frame)
        for frame in (0, 1, 10, -1):
            with self.assertRaises(ValueError):
                p.split(frame)
        self.assertEqual(before, (p.boundaries, [(s.id, s.note) for s in p.shots]))

    def test_one_frame_shots_and_generated_edit_sequence(self):
        p = Project.new(80)
        rng = random.Random(52)
        for _ in range(300):
            choices = [n for n in range(1, 80) if n not in p.boundaries]
            if choices and (len(p.shots) == 1 or rng.random() < .6):
                p.split(rng.choice(choices))
            else:
                p.merge(rng.randrange(len(p.shots) - 1))
            p.validate()
            self.assertEqual([n for row in range(len(p.shots)) for n in range(*p.interval(row))], list(range(80)))
            for frame in range(80):
                row = p.shot_at(frame)
                self.assertTrue(p.boundaries[row] <= frame < p.boundaries[row + 1])


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        source = self.root / "source.mp4"
        source.write_bytes(b"source fixture -- never overwritten")
        stat = source.stat()
        self.index = FrameIndex(str(source), stat.st_size, stat.st_mtime_ns, "1/24",
                                list(range(120)), [1] * 120, [0, 48, 96], True,
                                {"width": 16, "height": 16, "audio_tracks": 0}, 0)
        self.path = self.root / "project.shotlab"
        self.store = ProjectStore.create(self.path, self.index)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_media_relocation_and_edits_rollback_together(self):
        project, index, settings = self.store.load()
        project.shots[0].note = "未提交的笔记"
        self.store.connection.execute("CREATE TRIGGER reject_edit BEFORE INSERT ON shots BEGIN SELECT RAISE(ABORT, 'test interruption'); END")
        self.store.connection.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.save(project, settings, replace(index, path=str(self.root / "new-location.mp4")))
        restored, saved_index, _ = self.store.load()
        self.assertEqual(saved_index.path, index.path)
        self.assertEqual(restored.shots[0].note, "")
        self.assertEqual(self.store.revision, 0)

    def test_save_reopen_and_backup(self):
        p, index, settings = self.store.load()
        p.split(20)
        p.shots[0].note, p.shots[1].note = "甲\n多行", "乙"
        p.move_boundary(1, 25)
        p.current_frame = 46
        self.store.save(p, settings)
        self.store.backup(self.root / "backup.shotlab", p, settings)
        for path in [self.path, self.root / "backup.shotlab"]:
            other = ProjectStore(path)
            restored, source, _ = other.load()
            self.assertEqual(p, restored)
            self.assertEqual(source.pts, index.pts)
            other.close()

    def test_unicode_project_path_and_note(self):
        path = self.root / "镜头 分析 日本語.shotlab"
        store = ProjectStore.create(path, self.index)
        project, _, settings = store.load()
        project.shots[0].note = "观察 / 日本語 / emoji 🎬"
        store.save(project, settings)
        store.close()
        reopened = ProjectStore(path)
        self.assertEqual(reopened.load()[0].shots[0].note, project.shots[0].note)
        reopened.close()

    def test_concurrent_save_is_rejected_without_loss(self):
        other = ProjectStore(self.path)
        p, _, settings = self.store.load()
        q, _, qsettings = other.load()
        p.shots[0].note = "first writer"
        self.store.save(p, settings)
        q.shots[0].note = "unsaved second writer"
        with self.assertRaises(SaveConflict):
            other.save(q, qsettings)
        self.assertEqual(q.shots[0].note, "unsaved second writer")
        self.assertEqual(self.store.load()[0].shots[0].note, "first writer")
        other.close()

    def test_failed_sql_save_rolls_back_revision_and_shots(self):
        p, _, settings = self.store.load()
        self.store.connection.execute("CREATE TRIGGER fail_insert BEFORE INSERT ON shots BEGIN SELECT RAISE(ABORT,'injected failure'); END")
        p.split(30)
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.save(p, settings)
        restored, _, _ = self.store.load()
        self.assertEqual(restored.boundaries, [0, 120])
        self.assertEqual(self.store.revision, 0)

    def test_abrupt_exit_during_transaction_recovers_previous_commit(self):
        code = "import sqlite3,sys,os; c=sqlite3.connect(sys.argv[1]); c.execute('BEGIN IMMEDIATE'); c.execute('DELETE FROM shots'); os._exit(55)"
        result = subprocess.run([sys.executable, "-c", code, str(self.path)])
        self.assertEqual(result.returncode, 55)
        restored, _, _ = self.store.load()
        self.assertEqual(restored.boundaries, [0, 120])

    def test_source_and_existing_project_cannot_be_overwritten(self):
        project, _, settings = self.store.load()
        for path in [Path(self.index.path), self.path]:
            with self.assertRaises(FileExistsError):
                ProjectStore.create(path, self.index)
            with self.assertRaises(FileExistsError):
                self.store.backup(path, project, settings)
        self.index.verify_source()

    def test_backup_preserves_unsaved_memory_without_overwriting_original(self):
        project, _, settings = self.store.load()
        project.split(20)
        project.shots[1].note = "尚未提交的笔记"
        path = self.root / "unsaved-copy.shotlab"
        self.store.backup(path, project, settings)
        copy = ProjectStore(path)
        self.assertEqual(copy.load()[0], project)
        self.assertEqual(self.store.load()[0].boundaries, [0, 120])
        copy.close()

    def test_partial_index_cannot_claim_full_project(self):
        self.index.complete = False
        with self.assertRaises(ValueError):
            ProjectStore.create(self.root / "partial.shotlab", self.index)

    def test_newer_schema_is_left_unchanged(self):
        self.store.connection.execute("PRAGMA user_version=99")
        with self.assertRaises(ValueError):
            ProjectStore(self.path)
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM shots").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
