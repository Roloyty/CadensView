"""声音统计（第一档，只用 ffmpeg）：响度、静音、切点处的声音、跨集音频指纹。"""

import csv
import json
import threading
import unittest
from concurrent.futures import CancelledError
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from shotlab import audio
from shotlab.rpc import METHODS, Session, _audio_match
from tests.anime_fixtures import anime_clip
from tests.sound_fixtures import FFMPEG, cut_clip, episode
from tests.test_motion_slit import run_job
from tests.test_rpc import json_free_result


def _project(directory, source, *cuts):
    session = Session()
    session.build_index(str(source))
    session.new_project()
    session.apply_detection(json_free_result(directory))
    for frame in cuts:
        session._edit("拆分", mutate=lambda frame=frame: session.project.split(frame))
    return session


class MeterMathTests(unittest.TestCase):
    def test_gating_follows_bs1770(self):
        self.assertAlmostEqual(audio.gated_loudness([-20.0] * 10), -20.0, places=6)
        # 绝对门限 -70 LUFS 以下的块不计入
        self.assertAlmostEqual(audio.gated_loudness([-20.0] * 10 + [-90.0] * 50), -20.0, places=6)
        # 相对门限：比门内均值低 10 LU 以上的块不计入
        self.assertAlmostEqual(audio.gated_loudness([-20.0] * 10 + [-45.0] * 10), -20.0, places=6)
        self.assertIsNone(audio.gated_loudness([-80.0, -120.0]))

    def test_loudness_range_is_p95_minus_p10(self):
        values = np.linspace(-30, -10, 201)
        expected = np.percentile(values, 95) - np.percentile(values, 10)
        self.assertAlmostEqual(audio.loudness_range(values), expected, places=6)
        self.assertIsNone(audio.loudness_range([-90.0]))

    def test_silence_needs_minimum_duration(self):
        hop = 0.02
        levels = np.array([-20] * 10 + [-60] * 5 + [-20] * 10 + [-60] * 20 + [-20] * 5, dtype=float)
        mask = audio.silence_mask(levels, hop, threshold=-50, minimum=0.3)
        self.assertFalse(mask[10:15].any())   # 0.1 s 太短
        self.assertTrue(mask[25:45].all())    # 0.4 s 算静音
        self.assertEqual(int(mask.sum()), 20)


class FingerprintMatchTests(unittest.TestCase):
    def test_shared_stretch_is_found_with_bit_noise(self):
        rng = np.random.default_rng(3)
        cue = rng.integers(0, 2 ** 32, 200, dtype=np.uint64).astype(np.uint32)
        noisy = cue ^ (np.uint32(1) << rng.integers(0, 32, 200).astype(np.uint32))  # 每项翻 1 位
        a = np.concatenate([cue, rng.integers(0, 2 ** 32, 100, dtype=np.uint64).astype(np.uint32)])
        b = np.concatenate([rng.integers(0, 2 ** 32, 60, dtype=np.uint64).astype(np.uint32), noisy])
        found = audio.match_fingerprints(a, b)
        self.assertEqual(len(found), 1)
        self.assertLessEqual(abs(found[0]["a"][0] - 0), 3)
        self.assertLessEqual(abs(found[0]["b"][0] - 60), 3)
        self.assertAlmostEqual(found[0]["seconds"], 200 * audio.FP_ITEM_S, delta=1.0)

    def test_unrelated_and_constant_audio_do_not_match(self):
        rng = np.random.default_rng(4)
        a = rng.integers(0, 2 ** 32, 400, dtype=np.uint64).astype(np.uint32)
        b = rng.integers(0, 2 ** 32, 400, dtype=np.uint64).astype(np.uint32)
        self.assertEqual(audio.match_fingerprints(a, b), [])
        silence = np.full(400, 0x12345678, dtype=np.uint32)  # 静音段指纹恒定，不算共同段落
        self.assertEqual(audio.match_fingerprints(silence, silence.copy()), [])

    def test_pack_roundtrip(self):
        class Index:
            size, mtime_ns = 10, 20
        values = np.arange(1000, dtype=np.uint32) * 7919
        packed = audio.pack_fingerprint(values, 0.25, Index())
        self.assertTrue(np.array_equal(audio.unpack_fingerprint(packed), values))
        self.assertEqual(packed["offset_s"], 0.25)
        json.dumps(packed)


@unittest.skipUnless(FFMPEG.exists(), "tools/ffmpeg.exe missing")
class SoundAtCutsTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.session = _project(self.root, cut_clip(self.root / "cuts.mkv"), 96, 168)
        self.addCleanup(self.session.close_current_store)

    def test_shots_cuts_and_summary(self):
        done = run_job(self.session, "audio.run", {})
        self.assertEqual(done["phase"], "done", done)
        result = done["result"]
        shots, cuts = result["shots"], result["cuts"]
        self.assertEqual(len(shots), 3)
        self.assertEqual([c["frame"] for c in cuts], [96, 168])
        # 第一个切点：声音在画面切换前 0.5 秒就变了（J-cut）
        first = cuts[0]
        self.assertEqual(first["relation"], "J", first)
        self.assertAlmostEqual(first["sound_offset_s"], -0.5, delta=0.1)
        self.assertFalse(first["in_silence"])
        # 第二个切点：噪声在切点上戛然而止，进入静音
        second = cuts[1]
        self.assertEqual(second["relation"], "sync", second)
        self.assertTrue(second["in_silence"])
        self.assertLess(second["loudness_jump_lu"], -20)
        # 第三个镜头 3 秒里有 1 秒静音
        self.assertAlmostEqual(shots[2]["silence_share"], 1 / 3, delta=0.05)
        self.assertEqual(shots[0]["silence_share"], 0.0)
        for shot in shots:
            self.assertIsNotNone(shot["loudness_lufs"])
        self.assertIsNotNone(shots[0]["lra_lu"])  # 4 秒镜头有 LRA
        summary = result["summary"]
        self.assertEqual(summary["cuts"], 2)
        self.assertEqual(summary["relation_counts"], {"sync": 1, "J": 1, "L": 0, "none": 0})
        self.assertAlmostEqual(summary["cuts_in_silence_share"], 0.5)
        self.assertAlmostEqual(summary["silence_share"], 0.1, delta=0.02)
        self.assertTrue(result["curve"])
        self.assertIn("audio_fingerprint", self.session.settings_snapshot())
        self.assertTrue(self.session.dirty)

    def test_whole_track_loudness_matches_ffmpeg_meter(self):
        import re
        import subprocess
        source = self.root / "cuts.mkv"
        log = subprocess.run([str(FFMPEG), "-nostdin", "-hide_banner", "-i", str(source), "-map", "0:a",
                              "-af", "ebur128", "-f", "null", "-"], capture_output=True, text=True,
                             encoding="utf-8", errors="replace").stderr
        reference = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", log)[-1])
        track = self.session.load_soundtrack()
        self.assertAlmostEqual(audio.gated_loudness(track.momentary), reference, delta=0.2)

    def test_scope_and_staleness(self):
        ids = [shot["id"] for shot in self.session.state()["shots"]]
        run_job(self.session, "audio.run", {"shot_ids": ids[1:]})
        result = METHODS["audio.get"](self.session, {"shot_ids": ids[1:]})
        self.assertEqual(len(result["shots"]), 2)
        self.assertEqual(len(result["cuts"]), 1)  # 只有选中镜头之间的切点
        with self.assertRaises(ValueError):
            METHODS["audio.get"](self.session, {})
        self.session._edit("拆分", mutate=lambda: self.session.project.split(200))
        with self.assertRaisesRegex(ValueError, "声音分析"):  # 编辑后结果作废
            METHODS["audio.get"](self.session, {"shot_ids": ids[1:]})

    def test_research_export_carries_sound_columns(self):
        run_job(self.session, "audio.run", {})
        out = self.root / "research.csv"
        METHODS["research.export"](self.session, {"path": str(out)})
        with out.open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["cut_sound_relation"], "")  # 第一个镜头前没有切点
        self.assertEqual(rows[1]["cut_sound_relation"], "J")
        self.assertEqual(rows[2]["cut_in_silence"], "True")
        self.assertNotEqual(rows[0]["loudness_lufs"], "")
        method = json.loads(out.with_suffix(".method.json").read_text(encoding="utf-8"))
        self.assertIn("loudness", method["annotation_layers"]["audio"])

    def test_cancel_leaves_no_result(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(CancelledError):
            self.session.audio_run(self.session.analysis_scope({}), cancel=cancel)
        self.assertIsNone(self.session.audio)


@unittest.skipUnless(FFMPEG.exists(), "tools/ffmpeg.exe missing")
class SilentVideoTests(unittest.TestCase):
    def test_video_without_audio_is_reported(self):
        with TemporaryDirectory() as directory:
            session = _project(directory, anime_clip(Path(directory) / "anime.mkv"), 36)
            try:
                done = run_job(session, "audio.run", {})
                self.assertEqual(done["phase"], "error")
                self.assertIn("没有音轨", done["message"])
            finally:
                session.close_current_store()


@unittest.skipUnless(FFMPEG.exists(), "tools/ffmpeg.exe missing")
class EpisodeMatchTests(unittest.TestCase):
    def test_shared_cue_becomes_region_suggestion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            # 第 1 集：共同段落（OP）在 0–20 秒；第 2 集：在 15–35 秒（片尾）
            first = episode(root / "ep1.mkv", [("cue", 20, 0), ("bed", 10, 2)])
            second = episode(root / "ep2.mkv", [("bed", 15, 3), ("cue", 20, 0), ("bed", 5, 4)])
            one = _project(root, first, 80)
            try:
                run_job(one, "audio.run", {})
                METHODS["project.save_as"](one, {"path": str(root / "ep1.shotlab")})
            finally:
                one.close_current_store()
            # 第 2 集没做过声音分析：比对时自己算指纹，并从第 1 集项目里读已存的指纹
            two = _project(root, second, 120)
            try:
                done = run_job(two, "audio.match", {"paths": [str(root / "ep1.shotlab")]})
                self.assertEqual(done["phase"], "done", done)
                result = done["result"]
                self.assertEqual(result["episodes"][0]["fingerprint"], "stored")
                [suggestion] = result["suggestions"]
                self.assertEqual(suggestion["label"], "ED")
                self.assertAlmostEqual(suggestion["start_s"], 15, delta=1.5)
                self.assertAlmostEqual(suggestion["end_s"], 35, delta=1.5)
                self.assertAlmostEqual(suggestion["start_frame"], 120, delta=12)
                self.assertTrue(two.audio_fingerprint)
                METHODS["project.save_as"](two, {"path": str(root / "ep2.shotlab")})
            finally:
                two.close_current_store()
            # 反过来：第 1 集视角下是 OP
            one = Session()
            try:
                one.open_project(str(root / "ep1.shotlab"))
                # 自己的项目混在列表里会被跳过
                [suggestion] = _audio_match(one, {"paths": [str(root / "ep2.shotlab"),
                                                            str(root / "ep1.shotlab")]})["suggestions"]
                self.assertEqual(suggestion["label"], "OP")
                self.assertAlmostEqual(suggestion["start_s"], 0, delta=1.5)
            finally:
                one.close_current_store()


if __name__ == "__main__":
    unittest.main()
