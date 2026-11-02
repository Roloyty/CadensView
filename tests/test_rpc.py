"""Sidecar session semantics and the stdio JSON-RPC protocol."""

from fractions import Fraction
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import av
import numpy as np

from shotlab.rpc import Session, serve
from t0.media import FrameIndex


def video(path, frames=48):
    """The synthetic FFV1 fixture from the colour tests, reused for indexing."""
    with av.open(str(path), 'w') as output:
        stream = output.add_stream('ffv1', rate=24)
        stream.width, stream.height, stream.pix_fmt = 96, 54, 'yuv444p'
        for number in range(frames):
            array = np.zeros((54, 96, 3), np.uint8)
            array[:] = [255, 0, 0] if number % 24 < 12 else [0, 0, 255]
            frame = av.VideoFrame.from_ndarray(array, format='rgb24')
            frame.pts = number
            frame.time_base = Fraction(1, 24)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return path


def fake_index(path, frames=10):
    stat = Path(path).stat()
    return FrameIndex(str(path), stat.st_size, stat.st_mtime_ns, "1/24", list(range(frames)), [1] * frames,
                      [0], True, {"duration_us": frames * 41666}, 0.0)


def exchange(lines, session=None):
    sink = io.StringIO()
    serve(io.StringIO("\n".join(json.dumps(line) for line in lines) + "\n"), sink, session)
    replies = [json.loads(line) for line in sink.getvalue().splitlines()]
    events = [r for r in replies if "event" in r]
    responses = [r for r in replies if "id" in r]
    return responses, events


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.video_path = video(Path(self.directory.name) / "clip.mkv")

    def test_index_new_apply_detection_and_save_roundtrip(self):
        session = Session()
        built = session.build_index(str(self.video_path))
        self.assertEqual(built["index"]["complete"], True)
        self.assertGreaterEqual(built["index"]["metadata"]["duration_us"], 1900000)

        state = session.new_project()
        self.assertEqual((state["pending_start"], state["completed_count"]), (0, 0))
        self.assertEqual(len(state["shots"]), 1)

        result_dir = Path(self.directory.name) / "detected"
        result_dir.mkdir()
        (result_dir / "result.json").write_text(json.dumps(
            {"boundaries": [{"frame": 12}, {"frame": 36}]}), encoding="utf-8")
        state = session.apply_detection(str(result_dir))
        self.assertEqual([s["start"] for s in state["shots"]], [0, 12, 36])
        self.assertEqual(state["pending_start"], None)

        project_path = Path(self.directory.name) / "work.shotlab"
        session.save_as(str(project_path))
        session.close_project()
        state = session.open_project(str(project_path))
        self.assertEqual(len(state["shots"]), 3)
        self.assertEqual(session.project.boundaries, [0, 12, 36, 48])
        session.close_project()

    def test_edits_undo_redo_and_dirty_flag(self):
        session = Session()
        session.index = fake_index(self.video_path)
        session.new_project()
        session.apply_detection(json_free_result(self.directory.name))

        first = session.state()["shots"][0]["id"]
        session._edit("拆分", mutate=lambda: session.project.split(6))
        self.assertEqual(len(session.project.shots), 2)
        session.select(shot_id=first)
        session._edit("笔记", key=f"note:{first}",
                      mutate=lambda: session.project.set_note(first, "红色段"))
        self.assertEqual(session.project.shots[0].note, "红色段")
        self.assertTrue(session.dirty)

        session.project = session.history.restore(redo=False)
        self.assertEqual(session.project.shots[0].note, "")
        session.project = session.history.restore(redo=False)
        self.assertEqual(len(session.project.shots), 1)
        self.assertFalse(session.history.undo)


def json_free_result(directory):
    result_dir = Path(directory) / "detected2"
    result_dir.mkdir(exist_ok=True)
    (result_dir / "result.json").write_text(json.dumps({"boundaries": []}), encoding="utf-8")
    return str(result_dir)


class ProtocolTests(unittest.TestCase):
    def test_roundtrip_progress_errors_and_state(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        source = video(Path(directory.name) / "clip.mkv")
        detected = Path(directory.name) / "detected"
        detected.mkdir()
        (detected / "result.json").write_text(
            json.dumps({"boundaries": [{"frame": 24}]}), encoding="utf-8")

        responses, events = exchange([
            {"id": 1, "method": "ping"},
            {"id": 2, "method": "nope"},
            {"id": 3, "method": "index.build", "params": {"video_path": str(source)}},
            {"id": 4, "method": "project.new"},
            {"id": 5, "method": "edit.split", "params": {"frame": 20}},
            {"id": 6, "method": "project.apply_detection", "params": {"result_dir": str(detected)}},
            {"id": 7, "method": "edit.split", "params": {"frame": 30}},
            {"id": 8, "method": "edit.split", "params": {"frame": 30}},
            {"id": 9, "method": "edit.undo"},
            {"id": 10, "method": "project.state"},
        ])
        by_id = {r["id"]: r for r in responses}
        self.assertEqual(by_id[1]["result"], {"version": "0.1.0"})
        self.assertFalse(by_id[2]["ok"])
        self.assertTrue(by_id[3]["ok"])
        self.assertTrue(by_id[3]["result"]["index"]["complete"])
        self.assertEqual(by_id[4]["result"]["completed_count"], 0)
        # Splitting the pending region before detection must be rejected.
        self.assertFalse(by_id[5]["ok"])
        self.assertTrue(by_id[6]["ok"])
        self.assertEqual(by_id[6]["result"]["completed_count"], 2)
        self.assertTrue(by_id[7]["ok"])
        self.assertEqual(by_id[7]["result"]["shots"][2]["start"], 30)
        error = by_id[8]["error"]
        self.assertIn("boundary", error["message"].lower())
        self.assertTrue(by_id[9]["ok"])
        self.assertEqual(len(by_id[10]["result"]["shots"]), 2)


def selftest():
    """Offline smoke check used by `python -m shotlab.rpc --selftest`."""
    directory = TemporaryDirectory()
    try:
        source = video(Path(directory.name) / "selftest.mkv")
        session = Session()
        session.build_index(str(source))
        session.new_project()
        state = session.state()
        print(f"selftest ok: {state['frame_count']} frames indexed, protocol ready")
    finally:
        directory.cleanup()


if __name__ == "__main__":
    unittest.main()
