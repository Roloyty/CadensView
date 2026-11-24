"""2026-11-24 审查修复的回归测试。

覆盖：边车协议边界（请求形状、结果序列化、终态事件必达）、跨集音频比对降级、
字幕编码回退、NaN 入口、CLI 输出编码与导出默认值、拉片包与 VLM 接口报错。
"""

import argparse
import io
import json
import os
import subprocess
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from shotlab import audio, cli, lapian, rpc, vlm
from shotlab.compare import compare_projects
from shotlab.rpc import JOB_METHODS, JOB_RUNNERS, METHODS, Session, _audio_match, serve
from shotlab.study_formats import optional_asr_adapter
from tests.test_audio import _project
from tests.test_motion_slit import session_with_shots
from tests.test_rpc import exchange

ROOT = Path(__file__).resolve().parents[1]


def strict(line):
    """Parse the way the GUI does: serde_json refuses the NaN/Infinity tokens Python emits."""
    def reject(constant):
        raise ValueError(f"non-finite token {constant}")
    return json.loads(line, parse_constant=reject)


class RawSink:
    """serve() 写出的原始行；不像 json.loads 那样替 NaN 放行。"""

    def __init__(self):
        self._lines = []
        self._lock = threading.Lock()

    def reconfigure(self, **kwargs):
        pass

    def write(self, text):
        with self._lock:
            self._lines.extend(line for line in text.splitlines() if line.strip())

    def flush(self):
        pass

    def lines(self):
        with self._lock:
            return list(self._lines)


def job_outcome(session, name, params=None):
    """Run one job through the real protocol loop; every output line must be strict JSON."""
    sink = RawSink()
    request = json.dumps({"id": 1, "method": "job.start",
                          "params": {"method": name, "params": params or {}}})
    serve(io.StringIO(request + "\n"), sink, session)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        records = [strict(line) for line in sink.lines()]
        terminal = [record for record in records if record.get("phase") in ("done", "error")]
        if terminal:
            return terminal[-1]
        time.sleep(0.02)
    raise AssertionError(sink.lines())


class ProtocolBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.session = session_with_shots(self.directory.name)
        self.addCleanup(self.session.close_current_store)

    def test_malformed_requests_answer_with_errors_and_keep_the_loop_alive(self):
        # 改之前：第一行就让 serve() 抛 AttributeError，边车整个退出。
        requests = [[1], "text", {"method": 1}, {"id": 4, "method": "no.such.method"},
                    {"id": 5, "method": "tags.set", "params": "oops"},
                    {"id": 6, "method": "job.start", "params": {"method": []}},
                    {"id": 7, "method": "job.start", "params": {"method": "scale.run", "params": [1]}},
                    {"id": 8, "method": "ping"}]
        replies, _ = exchange(requests)
        self.assertEqual(len(replies), 8, replies)
        failed = [reply for reply in replies if not reply["ok"]]
        self.assertEqual(len(failed), 7, failed)
        self.assertEqual([reply["id"] for reply in failed if reply.get("id") is not None], [4, 5, 6, 7])
        self.assertEqual(replies[-1], {"id": 8, "ok": True, "result": {"version": "0.1.0"}})

    def test_job_result_with_numpy_and_non_finite_values_still_reports_done(self):
        # 改之前：发终态事件时 json.dumps 抛 TypeError，线程死掉，GUI 永久停在“任务进行中”。
        name = "test.unserializable"
        JOB_RUNNERS[name] = ("test", lambda s, p, c, g: {
            "int": np.int64(7), "flag": np.bool_(True), "single": np.float32(1.5), "curve": np.arange(3),
            "nan": float("nan"), "inf": float("inf"), "note": "字符串里写 NaN 没关系", "huge": 1 << 70})
        JOB_METHODS.add(name)
        self.addCleanup(JOB_RUNNERS.pop, name, None)
        self.addCleanup(JOB_METHODS.discard, name)
        outcome = job_outcome(self.session, name)
        self.assertEqual(outcome["phase"], "done", outcome)
        self.assertEqual(outcome["result"], {
            "int": 7, "flag": True, "single": 1.5, "curve": [0, 1, 2], "nan": None, "inf": None,
            "note": "字符串里写 NaN 没关系", "huge": str(1 << 70)})

    def test_dump_leaves_ordinary_payloads_alone_and_repairs_the_rest(self):
        from shotlab.rpc import _dump, _json_safe
        ordinary = {"a": 1, "b": [1.5, "x", None, True], "c": {"d": "文本"}}
        self.assertEqual(_dump(ordinary), json.dumps(ordinary, ensure_ascii=False))
        repaired = strict(_dump({
            "i": np.int64(-3), "f": np.float32(0.25), "b": np.bool_(False), "arr": np.array([[1, 2]]),
            "set": {2, 1}, "tuple": (1, 2), "nan": float("nan"), "neg": float("-inf"),
            "path": Path("x/y"), "frac": Fraction(1, 3), "bytes": b"abc", 7: "整数键",
            "big": 1 << 80, "low": -(1 << 63), "high": (1 << 64) - 1}))
        self.assertEqual(repaired, {
            "i": -3, "f": 0.25, "b": False, "arr": [[1, 2]], "set": [1, 2], "tuple": [1, 2],
            "nan": None, "neg": None, "path": str(Path("x/y")), "frac": 1 / 3, "bytes": "abc", "7": "整数键",
            "big": str(1 << 80), "low": -(1 << 63), "high": (1 << 64) - 1})
        self.assertEqual(_json_safe("NaN"), "NaN")  # 字符串原样保留

    def test_sync_reply_with_a_set_is_still_json(self):
        handler = METHODS["ping"]
        METHODS["ping"] = lambda session, params, emit=None: {"value": {1}}
        self.addCleanup(METHODS.__setitem__, "ping", handler)
        replies, _ = exchange([{"id": 1, "method": "ping"}], self.session)
        self.assertEqual(replies, [{"id": 1, "ok": True, "result": {"value": [1]}}])


class AudioMatchTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.session = session_with_shots(self.directory.name)
        self.addCleanup(self.session.close_current_store)

    def test_unreadable_episode_is_one_error_row_not_a_failed_job(self):
        # 改之前：sqlite3.OperationalError 不在捕获列表里，整次比对中止且不带文件名。
        self.session.audio_fingerprint = audio.pack_fingerprint(
            np.arange(40, dtype=np.uint32), 0.0, self.session.index)
        notes = self.root / "notes.txt"
        notes.write_text("这不是项目文件", encoding="utf-8")
        result = _audio_match(self.session, {"paths": [str(self.root / "missing.shotlab"), str(notes)]})
        missing, corrupt = result["episodes"]
        self.assertIn("missing.shotlab", missing["error"])
        self.assertIn("notes.txt", corrupt["error"])
        self.assertEqual(result["suggestions"], [])
        with self.assertRaises(ValueError):
            _audio_match(self.session, {"paths": [123]})


class FingerprintRecoveryTests(unittest.TestCase):
    def test_corrupt_stored_fingerprint_is_recomputed_not_fatal(self):
        from tests.sound_fixtures import cut_clip
        with TemporaryDirectory() as directory:
            root = Path(directory)
            session = _project(root, cut_clip(root / "cut.mkv"), 96, 168)
            try:
                from tests.test_motion_slit import run_job
                self.assertEqual(run_job(session, "audio.run", {})["phase"], "done")
                self.assertTrue(session.audio_fingerprint.get("data"))
                session.audio_fingerprint["data"] = "!!!既不是 base64 也不是 zlib!!!"
                fingerprint, _ = rpc._own_fingerprint(session)
                self.assertGreater(len(fingerprint), 0)
                audio.unpack_fingerprint(session.audio_fingerprint)  # 已被重算结果覆盖
                with self.assertRaises(ValueError):
                    audio.unpack_fingerprint({"data": "AAAA"})
            finally:
                session.close_current_store()


class SubtitleEncodingTests(unittest.TestCase):
    def test_subtitles_saved_as_gbk_or_utf16_load(self):
        # 改之前：只按 utf-8-sig 读，国内常见的 GBK 字幕直接 UnicodeDecodeError。
        with TemporaryDirectory() as directory:
            root = Path(directory)
            session = session_with_shots(directory)
            self.addCleanup(session.close_current_store)
            text = "1\n00:00:00,000 --> 00:00:01,000\n你好，世界\n"
            for encoding in ("gbk", "utf-16", "utf-8-sig"):
                with self.subTest(encoding=encoding):
                    path = root / f"{encoding}.srt"
                    path.write_bytes(text.encode(encoding))
                    cues = METHODS["subtitles.load"](session, {"path": str(path)})["entries"]
                    self.assertEqual([cue["text"] for cue in cues], ["你好，世界"])


class NonFiniteInputTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.session = session_with_shots(self.directory.name)
        self.addCleanup(self.session.close_current_store)

    def test_asr_confidence_must_be_a_finite_number(self):
        # json.loads 收得下 NaN 字面量；改之前它原样进入 study.asr_status 的返回值。
        meta = {"model": "m", "model_version": "1", "source": "s", "author": "a"}
        segments = json.loads('[{"start_s": 0, "end_s": 1, "text": "x", "confidence": NaN}]')
        with self.assertRaisesRegex(ValueError, "confidence"):
            optional_asr_adapter("whisperx", segments, meta)
        with self.assertRaisesRegex(ValueError, "confidence"):
            optional_asr_adapter("whisperx", [{"start_s": 0, "end_s": 1, "confidence": "高"}], meta)
        strict(json.dumps(optional_asr_adapter(
            "whisperx", [{"start_s": 0, "end_s": 1, "text": "x", "confidence": 0.9}], meta)))

    def test_narrative_compare_ignores_non_finite_numbers_from_other_projects(self):
        other_home = TemporaryDirectory()
        self.addCleanup(other_home.cleanup)
        other = session_with_shots(other_home.name)
        other.narrative = {"scenes": [], "curve_points": [{"time": 1, "intensity": float("nan"), "valence": 0},
                                                         {"time": 2, "intensity": 40, "valence": 5}]}
        path = self.root / "other.shotlab"
        other.save_as(str(path))
        other.close_current_store()
        film = lapian.compare_narratives([str(path)])["films"][0]
        # 两个点的 time 都有限：计 2 个曲线点；NaN 的 intensity 不进均值，valence [0, 5] 照常。
        self.assertEqual((film["curve_points"], film["mean_intensity"], film["mean_valence"]), (2, 40.0, 2.5))


class ProjectOpenTests(unittest.TestCase):
    def test_open_project_names_the_missing_source_video(self):
        # 改之前：裸的 FileNotFoundError “[WinError 2] 系统找不到指定的文件。”
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "work").mkdir()
            session = session_with_shots(root / "work")
            project = root / "p.shotlab"
            session.save_as(str(project))
            Path(session.index.path).rename(root / "moved.mkv")
            session.close_current_store()
            with self.assertRaisesRegex(ValueError, "找不到源视频"):
                Session().open_project(str(project))


class CliTests(unittest.TestCase):
    def test_output_survives_a_gbk_stdout(self):
        # 改之前：exit 2 + UnicodeEncodeError，连自己的错误 JSON 都打不出来。
        with TemporaryDirectory() as directory:
            root = Path(directory)
            session = session_with_shots(root)
            session.project.set_note(session.project.shots[0].id, "开场镜头 \U0001f3a5")
            project = root / "p.shotlab"
            session.save_as(str(project))
            session.close_current_store()
            env = dict(os.environ, PYTHONIOENCODING="gbk", PYTHONUTF8="0", PYTHONWARNINGS="ignore")
            completed = subprocess.run([sys.executable, "-m", "shotlab.cli", "shots", str(project)],
                                       cwd=ROOT, env=env, capture_output=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", "replace"))
            payload = json.loads(completed.stdout.decode("utf-8"))
            self.assertEqual(payload["shots"][0]["note"], "开场镜头 \U0001f3a5")

    def test_export_width_defaults_match_the_gui_and_mcp(self):
        # 改之前：stills 默认 160 像素宽（GUI/MCP 是 480），拉片长图默认 900（GUI/MCP 是 1600）。
        default = argparse.Namespace(width=None)
        self.assertEqual([cli._export_width(default, kind) for kind in (160, 480, 1600)], [160, 480, 1600])
        self.assertEqual(cli._export_width(argparse.Namespace(width=640), 480), 640)
        with self.assertRaises(cli.CliError):
            cli._export_width(argparse.Namespace(width=0), 480)


class LapianHardeningTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.session = session_with_shots(self.directory.name)
        self.addCleanup(self.session.close_current_store)

    def test_duplicate_or_multiline_scene_ids_are_normalized(self):
        payload = {"filmIdentity": lapian.build_identity(self.session), "scenes": [
            {"id": "s1", "start_shot": 1, "end_shot": 1, "title": "甲"},
            {"id": "s1", "start_shot": 2, "end_shot": 2, "title": "乙"},
            {"id": "bad\nid", "start_shot": 3, "end_shot": 3, "title": "丙"}]}
        narrative, _warnings, _preview = lapian.normalize_analysis(self.session, payload)
        ids = [scene["id"] for scene in narrative["scenes"]]
        self.assertEqual(ids, ["s1", "s1-2", "badid"])
        again = lapian.normalize_analysis(self.session, payload)[0]
        self.assertEqual([scene["id"] for scene in again["scenes"]], ids)  # 确定性：重复导入幂等

    def test_reexport_into_the_same_folder_drops_stale_images(self):
        out = self.root / "pkg"
        (out / "sheets").mkdir(parents=True)
        (out / "frames").mkdir()
        stale_sheet = out / "sheets" / "sheet-99.jpg"
        stale_sheet.write_bytes(b"old")
        keep = out / "frames" / "notes.txt"
        keep.write_text("x", encoding="utf-8")
        lapian.export_package(self.session, {"out_dir": str(out), "mode": "sheets"})
        self.assertFalse(stale_sheet.exists())
        self.assertEqual([p.name for p in (out / "sheets").glob("sheet-*.jpg")], ["sheet-01.jpg"])
        lapian.export_package(self.session, {"out_dir": str(out), "mode": "frames"})
        self.assertTrue(keep.exists())
        self.assertEqual(len(list((out / "frames").glob("shot-*.jpg"))), 3)

    def test_exported_srt_never_has_a_blank_line_inside_a_cue(self):
        self.session.subtitle_cues = [{"start_us": 0, "end_us": 1_000_000, "text": "第一行\n\n第二行"}]
        self.assertEqual(lapian.build_srt(self.session).split("\n"),
                         ["1", "00:00:00,000 --> 00:00:01,000", "第一行", "第二行", ""])

    def test_extract_json_gives_up_on_a_garbage_reply(self):
        # 旧实现对这种输入是 O(n²)：四万个起始括号各扫到结尾，要几十秒；现在按扫描步数封顶，约 2 秒。
        started = time.monotonic()
        with self.assertRaises(ValueError):
            lapian.extract_json("{" * 40000)
        self.assertLess(time.monotonic() - started, 8)
        self.assertEqual(lapian.extract_json('废话 {"a": {"b": 1}} 尾巴'), {"a": {"b": 1}})

    def test_prompt_tells_the_truth_about_a_scoped_package(self):
        rows = self.session.state()["shots"][1:]
        manifest = lapian.build_manifest(self.session, {"shot_ids": [row["id"] for row in rows]})
        prompt = lapian.build_prompt(manifest)
        self.assertIn("本次只分析 manifest.shots 列出的 2 个镜头", prompt)
        self.assertNotIn("范围 1-3", prompt)


class VlmHardeningTests(unittest.TestCase):
    def test_hand_edited_config_falls_back_to_defaults(self):
        with TemporaryDirectory() as directory:
            config_file = Path(directory) / "vlm.json"
            config_file.write_text(json.dumps({"concurrency": "2", "frames": 99, "timeout_s": 0,
                                               "image_width": "wide", "provider": 5}), encoding="utf-8")
            environment = {key: "" for key in os.environ if key.startswith("CADENSVIEW_VLM_")}
            with patch.object(vlm, "CONFIG_PATH", config_file), patch.dict(os.environ, environment):
                config = vlm.load_config()
        self.assertEqual((config["concurrency"], config["frames"]), (2, 6))
        self.assertEqual((config["timeout_s"], config["image_width"]), (1, vlm.DEFAULTS["image_width"]))
        self.assertEqual(config["provider"], "5")

    def test_http_failures_carry_the_provider_message(self):
        body = json.dumps({"error": {"code": "1302", "message": "您的账户已达到速率限制"}}).encode("utf-8")
        error = urllib.error.HTTPError("https://x/chat/completions", 429, "Too Many Requests", {}, io.BytesIO(body))
        with patch.object(urllib.request, "urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "HTTP 429.*速率限制"):
                vlm.request_chat("https://x/chat/completions", {}, {}, 5)
        self.assertEqual(vlm.chat_content({"choices": []}), ("", ""))
        self.assertEqual(vlm.chat_content({"choices": [{"message": {"content": None}}]}), ("", ""))
        self.assertEqual(vlm.chat_content({"model": "m", "choices": [
            {"message": {"content": [{"type": "text", "text": "好"}]}}]}), ("好", "m"))

    def test_last_attempt_error_is_described(self):
        class Backend:
            def annotate(self, shot, images):
                raise urllib.error.HTTPError("https://x", 401, "Unauthorized", {},
                                             io.BytesIO(b'{"error":{"message":"invalid api key"}}'))

            def retryable(self, error):
                return False

        with self.assertRaisesRegex(RuntimeError, "HTTP 401.*invalid api key"):
            vlm._call(Backend(), None, [])


class CompareTests(unittest.TestCase):
    def test_compare_names_the_project_it_cannot_read(self):
        # 改之前：裸的 "file is not a database"，多个项目里分不清是哪一个。
        with TemporaryDirectory() as directory:
            root = Path(directory)
            session = session_with_shots(root)
            first = root / "a.shotlab"
            session.save_as(str(first))
            session.close_current_store()
            plain = root / "b.shotlab"
            plain.write_text("不是 SQLite", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "b.shotlab"):
                compare_projects([str(first), str(plain)])
            with self.assertRaisesRegex(ValueError, "missing.shotlab"):
                compare_projects([str(first), str(root / "missing.shotlab")])


if __name__ == "__main__":
    unittest.main()
