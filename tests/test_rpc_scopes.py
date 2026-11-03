"""RPC 范围、字幕返回与存盘安全的回归测试。"""

import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from shotlab.rpc import METHODS, Session, serve

from tests.test_rpc import json_free_result, video


def completed_session(directory):
    session = Session()
    session.build_index(str(video(Path(directory) / "clip.mkv")))
    session.new_project()
    session.apply_detection(json_free_result(directory))
    return session


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.session = completed_session(self.directory.name)
        self.addCleanup(self.session.close_project)
        self.shots = self.session.state()["shots"]
        self.assertEqual(len(self.shots), 1)
        self.session._edit("拆分", mutate=lambda: self.session.project.split(12))
        self.session._edit("拆分", mutate=lambda: self.session.project.split(36))
        self.shots = self.session.state()["shots"]
        self.assertEqual([s["start"] for s in self.shots], [0, 12, 36])

    def test_stats_scope_keeps_only_selected_shots_and_no_fabricated_cuts(self):
        ids = [self.shots[0]["id"], self.shots[2]["id"]]
        result = METHODS["stats.run"](self.session, {"shot_ids": ids}, None)
        self.assertEqual([s["shot_id"] for s in result["shots"]], ids)
        self.assertEqual(result["overview"]["shot_count"], 2)
        self.assertEqual(result["editing"]["boundaries"], [],
                         "不连续选择之间不能凭空产生切镜")
        with self.assertRaises(ValueError):
            METHODS["stats.run"](self.session, {"shot_ids": []}, None)
        with self.assertRaises(ValueError):
            METHODS["stats.run"](self.session, {"shot_ids": ["missing"]}, None)

    def test_analysis_uses_same_scope_for_shot_statistics(self):
        ids = [self.shots[1]["id"]]
        result = METHODS["analysis.run"](self.session, {"shot_ids": ids}, None)
        self.assertEqual(result["shot_statistics"]["overview"]["shot_count"], 1)
        self.assertEqual([s["shot_id"] for s in result["shot_statistics"]["shots"]], ids)
        self.assertIn("video_color_timeline", result)

    def test_csv_export_respects_scope_and_order(self):
        ids = [self.shots[2]["id"], self.shots[0]["id"]]
        out = Path(self.directory.name) / "scoped.csv"
        result = METHODS["export.shots_csv"](self.session, {"path": str(out), "shot_ids": ids}, None)
        self.assertEqual(result["written"], 2)
        rows = out.read_text(encoding="utf-8-sig").splitlines()
        self.assertEqual(len(rows), 3)
        self.assertTrue(rows[1].startswith("1,"), rows)
        self.assertTrue(rows[2].startswith("3,"), rows)

    def test_contact_sheet_and_pdf_respect_scope(self):
        ids = [self.shots[1]["id"]]
        sheet = Path(self.directory.name) / "one.png"
        result = METHODS["export.contact_sheet"](self.session, {"path": str(sheet), "shot_ids": ids}, None)
        self.assertEqual(result["written"], 1)
        self.assertTrue(sheet.read_bytes().startswith(b"\x89PNG"))
        pdf = Path(self.directory.name) / "one.pdf"
        result = METHODS["export.pdf_sheet"](self.session, {"path": str(pdf), "shot_ids": ids}, None)
        self.assertEqual(result["written"], 1)
        self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))

    def test_subtitles_load_returns_entries_for_local_lookup(self):
        srt = Path(self.directory.name) / "sub.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n第一句\n\n"
                       "2\n00:00:01,500 --> 00:00:02,500\n第二句\n", encoding="utf-8")
        result = METHODS["subtitles.load"](self.session, {"path": str(srt)}, None)
        self.assertEqual(result["cues"], 2)
        self.assertEqual([e["text"] for e in result["entries"]], ["第一句", "第二句"])
        self.assertEqual(result["entries"][1]["start_us"], 1_500_000)
        at = METHODS["subtitles.at"](self.session, {"timestamp_us": 1_600_000}, None)
        self.assertEqual(at["text"], "第二句")

    def test_save_as_refuses_silent_overwrite_and_keeps_existing_file(self):
        target = Path(self.directory.name) / "existing.shotlab"
        target.write_bytes(b"do-not-destroy")
        with self.assertRaises(FileExistsError):
            self.session.save_as(str(target))
        self.assertEqual(target.read_bytes(), b"do-not-destroy")
        self.session.save_as(str(target), overwrite=True)
        self.assertTrue(target.read_bytes().startswith(b"SQLite format 3"))

    def test_new_project_clears_tags_scale_and_subtitles(self):
        first = self.shots[0]["id"]
        METHODS["tags.set"](self.session, {"shot_id": first, "tags": ["对话"]}, None)
        self.session.scale_labels[first] = {"label": "MS", "label_zh": "中景", "start": 0, "end": 12}
        self.session.subtitle_cues = [{"start_us": 0, "end_us": 1, "text": "x"}]
        self.session.new_project()
        self.assertEqual(self.session.shot_tags, {})
        self.assertEqual(self.session.scale_labels, {})
        self.assertEqual(self.session.subtitle_cues, [])


class JobOrderTests(unittest.TestCase):
    def test_ack_is_written_before_worker_events(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        source = video(Path(directory.name) / "clip.mkv")
        lines = [{"id": 1, "method": "job.start",
                  "params": {"method": "index.build", "params": {"video_path": str(source)}}}]
        sink = io.StringIO()
        serve(io.StringIO("\n".join(json.dumps(line) for line in lines) + "\n"), sink)
        records = [json.loads(line) for line in sink.getvalue().splitlines()]
        self.assertTrue(records and records[0].get("id") == 1 and records[0].get("ok"))
        import time
        deadline = time.time() + 10
        while time.time() < deadline:
            records = [json.loads(line) for line in sink.getvalue().splitlines() if line.strip()]
            if any(r.get("phase") == "done" for r in records):
                break
            time.sleep(0.05)
        self.assertTrue(any(r.get("phase") == "done" for r in records), records)


if __name__ == "__main__":
    unittest.main()
