"""freder/cinemetrics 功能移植：运动量、扫描条、静帧。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import io
import json
import time
import unittest

from shotlab.motion import analyze_motion, write_spectrum
from shotlab.rpc import Session
from shotlab.slit import build_sheet, export_stills
from shotlab.statistics import AnalysisScope

from shotlab.rpc import serve
from tests.test_rpc import json_free_result, video


def run_job(session, name, params):
    sink = io.StringIO()
    request = {"id": 1, "method": "job.start", "params": {"method": name, "params": params}}
    serve(io.StringIO(json.dumps(request) + "\n"), sink, session)
    deadline = time.time() + 30
    while time.time() < deadline:
        records = [json.loads(line) for line in sink.getvalue().splitlines() if line.strip()]
        done = [r for r in records if r.get("phase") in ("done", "error")]
        if done:
            return done[-1]
        time.sleep(0.05)
    raise AssertionError(sink.getvalue())


def session_with_shots(directory):
    session = Session()
    session.build_index(str(video(Path(directory) / "clip.mkv")))
    session.new_project()
    session.apply_detection(json_free_result(directory))
    session._edit("拆分", mutate=lambda: session.project.split(12))
    session._edit("拆分", mutate=lambda: session.project.split(36))
    return session


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.session = session_with_shots(self.directory.name)

    def test_motion_excludes_first_frame_and_marks_cuts(self):
        ids = [s["id"] for s in self.session.state()["shots"]]
        scope = AnalysisScope(kind="selected_shots", shot_ids=tuple(ids))
        result = analyze_motion(self.session.project, self.session.index, scope)
        self.assertEqual(len(result["shots"]), 3)
        by_number = {row["shot_number"]: row for row in result["shots"]}
        # 第一个镜头是纯红，首帧被排除，镜头均值应为 0
        self.assertEqual(by_number[1]["motion"], 0.0)
        # 中间镜头内部有一次红→蓝切换，均值必须大于 0
        self.assertGreater(by_number[2]["motion"], 0.0)
        self.assertGreater(len(result["samples"]), 0)
        self.assertTrue(all(0.0 <= s["motion"] <= 1.0 for s in result["samples"]))

    def test_spectrum_png(self):
        result = analyze_motion(self.session.project, self.session.index)
        path = Path(self.directory.name) / "motion.png"
        written = write_spectrum(result, path, width=640, height=200)
        self.assertEqual(written["path"], str(path.resolve()))
        self.assertTrue(path.read_bytes().startswith(b"\x89PNG"))

    def test_rpc_motion_cache_rejects_changed_scope(self):
        from shotlab.rpc import METHODS
        shots = self.session.state()["shots"]
        first = {"shot_ids": [shots[0]["id"]]}
        result = self.session.motion_run(self.session.analysis_scope(first))
        self.assertEqual(len(result["shots"]), 1)
        self.assertEqual(METHODS["motion.get"](self.session, first, None)["shots"][0]["shot_number"], 1)
        with self.assertRaises(ValueError):
            METHODS["motion.get"](self.session, {"shot_ids": [shots[1]["id"]]}, None)


class JobTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.session = session_with_shots(self.directory.name)
        self.addCleanup(self.session.close_project)

    def test_motion_job_returns_scoped_result(self):
        shots = self.session.state()["shots"]
        event = run_job(self.session, "motion.run", {"shot_ids": [shots[1]["id"]]})
        self.assertEqual(event["phase"], "done", event)
        self.assertEqual(len(event["result"]["shots"]), 1)

    def test_slit_job_writes_sheet(self):
        path = Path(self.directory.name) / "job-slits.png"
        event = run_job(self.session, "slit.run", {"path": str(path), "width": 24, "columns": 3})
        self.assertEqual(event["phase"], "done", event)
        self.assertEqual(event["result"]["written"], 3)
        self.assertTrue(path.exists())


class SlitTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.session = session_with_shots(self.directory.name)

    def test_slit_sheet_has_one_strip_per_shot(self):
        from shotlab.statistics import build_context
        shots = build_context(self.session.project, self.session.index).records
        path = Path(self.directory.name) / "slits.png"
        result = build_sheet(self.session.index, shots, path, width=32, columns=3, cell_height=48)
        self.assertEqual(result["written"], 3)
        self.assertTrue(path.read_bytes().startswith(b"\x89PNG"))

    def test_stills_are_evenly_named_and_written(self):
        folder = Path(self.directory.name) / "stills"
        result = export_stills(self.session.project, self.session.index, folder, count=10)
        self.assertEqual(result["written"], 10)
        names = sorted(p.name for p in folder.glob("*.jpg"))
        self.assertEqual(names[0], "still_0001.jpg")
        self.assertEqual(names[-1], "still_0010.jpg")


if __name__ == "__main__":
    unittest.main()
