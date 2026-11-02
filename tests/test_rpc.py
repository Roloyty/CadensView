"""Sidecar session semantics and the stdio JSON-RPC protocol."""

from fractions import Fraction
import io
import json
from pathlib import Path
import queue
from tempfile import TemporaryDirectory
import threading
import time
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
    # 与真实 build_index 的 metadata 契约保持一致（ProjectStore.create 依赖这些键）。
    metadata = {"container": "matroska", "codec": "ffv1", "width": 96, "height": 54,
                "pixel_format": "yuv444p", "average_rate": "24/1",
                "duration_us": frames * 41666, "audio_tracks": 0, "subtitle_tracks": 0}
    return FrameIndex(str(path), stat.st_size, stat.st_mtime_ns, "1/24", list(range(frames)), [1] * frames,
                      [0], True, metadata, 0.0)


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


class RegressionTests(unittest.TestCase):
    """对抗式审查发现的问题：逐条固化，防回归。"""

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.video_path = video(Path(self.directory.name) / "clip.mkv")

    def test_select_does_not_mark_dirty(self):
        session = Session()
        session.index = fake_index(self.video_path)
        session.new_project()
        session.dirty = False
        state = session.select(frame=3)
        self.assertFalse(state["dirty"], "移动光标不应标记未保存")

    def test_stats_cache_invalidated_by_edits(self):
        session = Session()
        session.index = fake_index(self.video_path)
        session.new_project()
        session.apply_detection(json_free_result(self.directory.name))
        session.stats = {"stale": True}
        session._edit("拆分", mutate=lambda: session.project.split(4))
        self.assertIsNone(session.stats, "编辑后统计缓存必须失效")

    def test_build_index_releases_previous_store(self):
        """换片重建索引必须释放旧项目文件句柄（Windows 下不释放会锁文件）。"""
        session = Session()
        session.index = fake_index(self.video_path)
        session.new_project()
        project_path = Path(self.directory.name) / "keep.shotlab"
        session.save_as(str(project_path))
        # 重建索引（模拟"打开视频"换片）
        session.index = fake_index(self.video_path)
        session.close_project()
        project_path.unlink()  # 未释放句柄时 Windows 会抛 PermissionError
        self.assertFalse(project_path.exists())

    def test_save_as_switches_to_new_file(self):
        """另存为在已有项目文件时也必须工作（走 backup 副本路径）。"""
        session = Session()
        session.index = fake_index(self.video_path)
        session.new_project()
        first = Path(self.directory.name) / "first.shotlab"
        session.save_as(str(first))
        second = Path(self.directory.name) / "second.shotlab"
        state = session.save_as(str(second))
        self.assertTrue(second.exists())
        self.assertEqual(state["project_path"], str(second.resolve()))
        # 再保存应写入新文件
        session.select(frame=2)
        session.dirty = True
        session.save()
        session.close_project()
        reopened = Session()
        reopened.open_project(str(second))
        self.assertEqual(reopened.project.current_frame, 2)
        reopened.close_project()


class JobProtocolTests(unittest.TestCase):
    """job.start / job.cancel：进度事件、忙等保护、取消。"""

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.video_path = video(Path(self.directory.name) / "clip.mkv")
        self.sink = EventSink()
        self.stdin = LineIn()
        self.thread = threading.Thread(
            target=serve, args=(self.stdin, self.sink, Session()), daemon=True)
        self.thread.start()

    def tearDown(self):
        self.stdin.close()
        self.thread.join(timeout=5)

    def request(self, id, method, params=None):
        self.stdin.put(json.dumps({"id": id, "method": method, "params": params or {}}))
        reply = self.sink.wait_for(lambda lines: next(
            (l for l in lines if l.get("id") == id), None))
        return reply

    def test_index_build_job_progress_done_and_busy_gate(self):
        # 进度事件每 240 帧发一次，用小视频测不出来，这里用长视频。
        long_video = video(Path(self.directory.name) / "long.mkv", frames=2400)
        reply = self.request(1, "job.start", {
            "method": "index.build",
            "params": {"video_path": str(long_video)}})
        self.assertTrue(reply["ok"], reply)
        job_id = reply["result"]["job_id"]
        # 索引进行中：状态查询放行（BUSY_ALLOWED），编辑操作被拒。
        state_reply = self.request(2, "project.state")
        self.assertTrue(state_reply["ok"], "忙时读状态必须放行")
        done = self.sink.wait_for(lambda lines: next(
            (l for l in lines if l.get("job_id") == job_id and l.get("phase") == "done"), None),
            timeout=120)
        self.assertEqual(done["topic"], "index.build")
        self.assertTrue(done["result"]["index"]["complete"])
        progress = [l for l in self.sink.lines() if l.get("job_id") == job_id and l.get("phase") == "progress"]
        self.assertTrue(progress, "index.build 必须发出进度事件（对抗式审查修复点）")
        self.assertEqual(progress[0]["topic"], "index")

    def test_job_cancel_stops_index_build(self):
        # 大一点的视频保证来得及取消
        long_video = video(Path(self.directory.name) / "long.mkv", frames=2400)
        reply = self.request(1, "job.start", {
            "method": "index.build",
            "params": {"video_path": str(long_video)}})
        job_id = reply["result"]["job_id"]
        self.request(2, "job.cancel", {"job_id": job_id})
        outcome = self.sink.wait_for(lambda lines: next(
            (l for l in lines if l.get("job_id") == job_id and l.get("phase") in ("done", "error")), None),
            timeout=60)
        self.assertEqual(outcome["phase"], "error")
        self.assertIn("cancel", outcome["message"].lower())
        # 取消后会话不再忙
        reply = self.request(3, "project.state")
        self.assertTrue(reply["ok"])


class EventSink:
    """捕获 serve() 输出的行并支持等待条件。"""

    def __init__(self):
        self._lines = []
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def reconfigure(self, **kwargs):
        pass

    def write(self, text):
        for line in text.splitlines():
            if not line.strip():
                continue
            with self._cond:
                self._lines.append(json.loads(line))
                self._cond.notify_all()

    def flush(self):
        pass

    def lines(self):
        with self._lock:
            return list(self._lines)

    def wait_for(self, predicate, timeout=10):
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                found = predicate(list(self._lines))
                if found:
                    return found
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(f"等待超时；已收到 {self._lines}")
                self._cond.wait(remaining)


class LineIn:
    """阻塞式行输入，模拟 GUI 的 stdin 管道。"""

    def __init__(self):
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False

    def reconfigure(self, **kwargs):
        pass

    def put(self, line):
        self._queue.put(line)

    def close(self):
        self._queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get(timeout=60)
        if item is None:
            raise StopIteration
        return item


class ExportTests(unittest.TestCase):
    """export.shots_csv：中文字段名 + utf-8-sig 编码。"""

    def test_csv_export_with_chinese_headers(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        source = video(Path(directory.name) / "clip.mkv")
        from shotlab.rpc import Session, METHODS
        session = Session()
        session.build_index(str(source))
        session.new_project()
        result = METHODS["export.shots_csv"](session, {"path": str(Path(directory.name) / "out.csv")}, None)
        out = Path(directory.name) / "out.csv"
        self.assertTrue(out.exists())
        self.assertEqual(result["written"], len(session.project.shots))
        content = out.read_bytes()
        self.assertTrue(content.startswith(b'\xef\xbb\xbf'), "CSV 需要 utf-8-sig BOM（Excel 兼容）")
        text = content.decode("utf-8-sig")
        self.assertIn("镜头编号", text)
        self.assertIn("时长(秒)", text)
