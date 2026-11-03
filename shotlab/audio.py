"""Soundtrack statistics with the bundled ffmpeg only (no extra dependencies).

One decode pass yields three things:
  * the EBU R 128 meter log (ebur128 filter): momentary (400 ms) and short-term
    (3 s) loudness every 100 ms, from which per-Shot integrated loudness
    (ITU-R BS.1770 gating) and loudness range (EBU Tech 3342) are recomputed;
  * mono PCM at 11 025 Hz for silence runs and a spectral-change curve;
  * a Chromaprint fingerprint, stored with the project so other episodes can
    find shared openings/endings (the method of Jellyfin's Intro Skipper,
    reimplemented here).

Times are seconds from the first indexed video frame, the coordinate the GUI
and ffmpeg -ss use. Audio time 0 is the first decoded sample; the stream start
difference between audio and video is read from the container and applied.
"""

import base64
import re
import subprocess
import tempfile
import threading
import zlib
from concurrent.futures import CancelledError
from pathlib import Path

import numpy as np

from shotlab.statistics import AnalysisScope, build_context

FFMPEG = Path(__file__).resolve().parent.parent / "tools" / "ffmpeg.exe"

SAMPLE_RATE = 11025
HOP = 256                 # ~23 ms analysis hop
WINDOW = 1024
BANDS = 24                # log-spaced bands, 60 Hz .. Nyquist
BLOCK_S = 0.1             # ebur128 log period
MOMENTARY_S = 0.4
SHORT_TERM_S = 3.0
ABSOLUTE_GATE = -70.0
SILENCE_DB = -50.0        # dBFS RMS of the mono mix
SILENCE_MIN_S = 0.3
NOVELTY_HALF_S = 0.5      # spectrum compared over 0.5 s before vs after a point
CHANGE_DB = 6.0           # mean band-level change counted as a sound change
CUT_WINDOW_S = 1.0        # a cut's sound change is searched within +-1 s
SYNC_FRAMES = 2           # |offset| within 2 frames counts as synchronous
MAX_CURVE_POINTS = 3000

# Chromaprint (algorithm 1, the fpcalc default): one item per 4096/3 samples at 11 025 Hz.
FP_ITEM_S = 4096 / 3 / 11025
FP_MAX_BITS = 6           # item pairs differing in <= 6 of 32 bits match (Intro Skipper)
FP_MIN_S = 15.0
FP_MAX_GAP_S = 3.5
FP_SMOOTH = 9             # items (~1.1 s) averaged before the bit-error test
FP_COMMON = 20            # values this frequent (silence, tones) do not vote for an offset

_METER = re.compile(r"t:\s*([-\d.]+)\s+TARGET:.*?M:\s*(\S+)\s+S:\s*(\S+)")
_TIME = re.compile(r"time=(\d+):(\d+):([\d.]+)")


def _cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise CancelledError("声音分析已取消；项目数据未改动")


def _number(text):
    try:
        value = float(text)
    except ValueError:
        return -120.0
    return value if np.isfinite(value) else -120.0


def stream_offset(path, index):
    """(audio start - first video frame) in seconds, or None when the file has no audio."""
    import av
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        start = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
    return start - float(index.pts[0] * index.tb)


def decode(path, duration_s=None, cancel=None, progress=None, ffmpeg=FFMPEG):
    """Run the single ffmpeg pass; return raw features of the whole soundtrack."""
    with tempfile.TemporaryDirectory(prefix="cv-audio-") as folder:
        fingerprint_path = Path(folder) / "fp.bin"
        command = [str(ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "info", "-i", str(path),
                   "-map", "0:a:0", "-af", "ebur128=framelog=info", "-f", "null", "-",
                   "-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1",
                   "-map", "0:a:0", "-ac", "1", "-f", "chromaprint", "-fp_format", "raw",
                   str(fingerprint_path)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        meter, tail = [], []

        def read_log():
            for raw in iter(process.stderr.readline, b""):
                for line in raw.decode("utf-8", "replace").split("\r"):
                    hit = _METER.search(line)
                    if hit:
                        meter.append((_number(hit.group(2)), _number(hit.group(3))))
                        if progress is not None and duration_s:
                            progress(min(int(float(hit.group(1)) * 10), int(duration_s * 10)),
                                     max(1, int(duration_s * 10)))
                    elif line.strip():
                        tail.append(line.strip())
                        del tail[:-20]

        reader = threading.Thread(target=read_log, daemon=True)
        reader.start()
        chunks = []
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    process.kill()
                    break
                chunk = process.stdout.read(1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
            process.wait()
            reader.join()
        finally:
            if process.poll() is None:
                process.kill()
        _cancel(cancel)
        if process.returncode != 0:
            message = " | ".join(tail[-4:]) or f"exit {process.returncode}"
            raise RuntimeError(f"ffmpeg 解码音轨失败: {message}")
        fingerprint = (np.frombuffer(fingerprint_path.read_bytes(), dtype="<u4").copy()
                       if fingerprint_path.exists() else np.zeros(0, dtype=np.uint32))
    pcm = np.frombuffer(b"".join(chunks), dtype="<f4")
    if not meter or pcm.size == 0:
        raise RuntimeError("ffmpeg 没有产出音频数据")
    loudness = np.array(meter, dtype=np.float64)
    return dict(momentary=loudness[:, 0], short_term=loudness[:, 1], pcm=pcm, fingerprint=fingerprint)


def frame_features(pcm, sample_rate=SAMPLE_RATE):
    """Per-hop RMS level (dBFS) and spectral-change curve (dB) on the HOP grid."""
    count = max(1, (pcm.size - WINDOW) // HOP + 1) if pcm.size >= WINDOW else 1
    padded = pcm if pcm.size >= WINDOW else np.pad(pcm, (0, WINDOW - pcm.size))
    hop_frames = padded[:count * HOP].reshape(count, HOP)
    rms_db = 10 * np.log10(np.mean(hop_frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    rms_db = np.maximum(rms_db, -120.0)
    edges = np.geomspace(60, sample_rate / 2, BANDS + 1)
    bins = np.fft.rfftfreq(WINDOW, 1 / sample_rate)
    band_of = np.searchsorted(edges, bins, side="right") - 1
    onehot = np.zeros((bins.size, BANDS), dtype=np.float32)
    valid = np.flatnonzero((band_of >= 0) & (band_of < BANDS))
    onehot[valid, band_of[valid]] = 1.0
    window = np.hanning(WINDOW).astype(np.float32)
    levels = np.empty((count, BANDS), dtype=np.float32)
    step = 4096
    for start in range(0, count, step):
        stop = min(count, start + step)
        offsets = (np.arange(start, stop) * HOP)[:, None] + np.arange(WINDOW)[None, :]
        spectrum = np.abs(np.fft.rfft(padded[np.minimum(offsets, padded.size - 1)] * window, axis=1)) ** 2
        levels[start:stop] = 10 * np.log10(spectrum.astype(np.float32) @ onehot + 1e-12)
    floor = float(levels.max()) - 80.0 if levels.size else -120.0
    levels = np.maximum(levels, floor)
    half = max(1, round(NOVELTY_HALF_S * sample_rate / HOP))
    total = np.vstack([np.zeros((1, BANDS)), np.cumsum(levels, axis=0, dtype=np.float64)])
    novelty = np.zeros(count)
    if count > 2 * half:
        centre = np.arange(half, count - half + 1)
        before = (total[centre] - total[centre - half]) / half
        after = (total[centre + half] - total[centre]) / half
        novelty[centre] = np.sqrt(np.mean((after - before) ** 2, axis=1))
    # Spectral frame c is centred WINDOW/2 samples after hop c starts: re-index to hop time.
    lag = WINDOW // 2 // HOP
    novelty = np.concatenate([np.zeros(lag), novelty])[:count]
    return dict(rms_db=rms_db, novelty=novelty, hop_s=HOP / sample_rate)


def silence_mask(rms_db, hop_s, threshold=SILENCE_DB, minimum=SILENCE_MIN_S):
    """True for hops inside a silent run of at least `minimum` seconds."""
    quiet = rms_db < threshold
    mask = np.zeros_like(quiet)
    need = max(1, round(minimum / hop_s))
    start = None
    for i, value in enumerate(np.append(quiet, False)):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start >= need:
                mask[start:i] = True
            start = None
    return mask


def gated_loudness(values, relative=-10.0):
    """BS.1770 integrated loudness of momentary block loudness values (LUFS), or None."""
    values = np.asarray(values, dtype=np.float64)
    values = values[values > ABSOLUTE_GATE]
    if values.size == 0:
        return None
    energy = 10 ** ((values + 0.691) / 10)
    gate = -0.691 + 10 * np.log10(energy.mean()) + relative
    kept = energy[values > gate]
    return float(-0.691 + 10 * np.log10(kept.mean())) if kept.size else None


def loudness_range(values):
    """EBU Tech 3342 LRA from short-term loudness values (LU), or None."""
    values = np.asarray(values, dtype=np.float64)
    values = values[values > ABSOLUTE_GATE]
    if values.size < 2:
        return None
    energy = 10 ** (values / 10)
    gate = 10 * np.log10(energy.mean()) - 20
    kept = values[values > gate]
    if kept.size < 2:
        return None
    return float(np.percentile(kept, 95) - np.percentile(kept, 10))


class Soundtrack:
    """Decoded features of one source, addressed in seconds from the first video frame."""

    def __init__(self, raw, offset_s, frames=None):
        self.offset_s = float(offset_s)
        self.momentary = raw["momentary"]
        self.short_term = raw["short_term"]
        self.fingerprint = raw["fingerprint"]
        frames = frames if frames is not None else frame_features(raw["pcm"])
        self.rms_db = frames["rms_db"]
        self.novelty = frames["novelty"]
        self.hop_s = frames["hop_s"]
        self.silent = silence_mask(self.rms_db, self.hop_s)
        self.duration_s = raw["pcm"].size / SAMPLE_RATE

    def _blocks(self, values, length, start, end):
        """Meter values whose whole window [t-length, t] lies inside [start, end)."""
        ends = (np.arange(len(values)) + 1) * BLOCK_S + self.offset_s
        inside = (ends - length >= start - 1e-6) & (ends <= end + 1e-6)
        return values[inside]

    def _hops(self, start, end):
        first = max(0, int(np.floor((start - self.offset_s) / self.hop_s)))
        last = min(len(self.rms_db), int(np.ceil((end - self.offset_s) / self.hop_s)))
        return slice(first, max(first, last))

    def momentary_at(self, seconds):
        """Momentary loudness of the 400 ms block ending at `seconds` (LUFS)."""
        k = int(round((seconds - self.offset_s) / BLOCK_S)) - 1
        if 0 <= k < len(self.momentary):
            return max(float(self.momentary[k]), ABSOLUTE_GATE)
        return None

    def span(self, start, end):
        blocks = self._blocks(self.momentary, MOMENTARY_S, start, end)
        short = self._blocks(self.short_term, SHORT_TERM_S, start, end)
        hops = self._hops(start, end)
        silent = self.silent[hops]
        return dict(
            loudness_lufs=_round(gated_loudness(blocks)),
            loudness_max_lufs=_round(float(blocks.max())) if blocks.size and blocks.max() > ABSOLUTE_GATE else None,
            lra_lu=_round(loudness_range(short)) if end - start >= SHORT_TERM_S else None,
            silence_share=_round(float(silent.mean()), 4) if silent.size else None)

    def cut(self, seconds, fps):
        """Sound at a picture cut: silence, loudness jump, nearest spectral change."""
        hop = int(round((seconds - self.offset_s) / self.hop_s))
        in_silence = bool(self.silent[max(0, hop - 1):hop + 2].any()) if 0 <= hop < len(self.silent) else None
        before, after = self.momentary_at(seconds), self.momentary_at(seconds + MOMENTARY_S)
        jump = _round(after - before) if before is not None and after is not None else None
        reach = int(round(CUT_WINDOW_S / self.hop_s))
        lo, hi = max(0, hop - reach), min(len(self.novelty), hop + reach + 1)
        offset = change = None
        relation = "none"
        if hi > lo:
            peak = lo + int(np.argmax(self.novelty[lo:hi]))
            change = float(self.novelty[peak])
            if change >= CHANGE_DB:
                offset = (peak - hop) * self.hop_s
                tolerance = max(SYNC_FRAMES / fps if fps else 0.084, self.hop_s * 1.5)
                relation = "sync" if abs(offset) <= tolerance else ("J" if offset < 0 else "L")
        return dict(in_silence=in_silence, loudness_jump_lu=jump, sound_change_db=_round(change),
                    sound_offset_s=_round(offset, 3), relation=relation)

    def curve(self, duration_s):
        """Momentary loudness binned to at most MAX_CURVE_POINTS points."""
        times = (np.arange(len(self.momentary)) + 1) * BLOCK_S + self.offset_s
        keep = (times >= 0) & (times <= duration_s + BLOCK_S)
        times, values = times[keep], np.maximum(self.momentary[keep], ABSOLUTE_GATE)
        stride = max(1, int(np.ceil(len(values) / MAX_CURVE_POINTS)))
        return [dict(video_time_s=round(float(times[i]), 3),
                     loudness_lufs=round(float(values[i:i + stride].max()), 2))
                for i in range(0, len(values), stride)]


def _round(value, digits=2):
    return None if value is None else round(float(value), digits)


def _median(values):
    return float(np.median(values)) if values else None


def analyze_audio(project, index, soundtrack, scope=AnalysisScope()):
    """Per-Shot loudness/silence and per-cut sound relations for the Shots in scope."""
    context = build_context(project, index, scope)
    shots = context.records
    if not shots:
        raise ValueError("没有可分析的已完成镜头")
    fps = None
    rate = index.metadata.get("average_rate")
    try:
        from fractions import Fraction
        fps = float(Fraction(str(rate))) if rate else None
    except (ValueError, ZeroDivisionError):
        fps = None
    rows, cuts = [], []
    blocks, short = [], []
    for shot in shots:
        start, end = float(shot.start), float(shot.end)
        rows.append(dict(shot.to_dict(), **soundtrack.span(start, end)))
        blocks.append(soundtrack._blocks(soundtrack.momentary, MOMENTARY_S, start, end))
        short.append(soundtrack._blocks(soundtrack.short_term, SHORT_TERM_S, start, end))
    for left, right in zip(shots, shots[1:]):
        if left.end_frame_exclusive != right.start_frame:
            continue  # not adjacent inside the scope: no cut between them was selected
        cuts.append(dict(frame=right.start_frame, time_s=round(float(right.start), 4),
                         left_shot_id=left.shot_id, right_shot_id=right.shot_id,
                         **soundtrack.cut(float(right.start), fps)))
    covered = sum(float(s.duration) for s in shots)
    silent_s = sum((row["silence_share"] or 0) * row["duration_s"] for row in rows)
    relations = {key: sum(1 for c in cuts if c["relation"] == key) for key in ("sync", "J", "L", "none")}
    judged = [c for c in cuts if c["in_silence"] is not None]
    offsets = [abs(c["sound_offset_s"]) for c in cuts if c["sound_offset_s"] is not None]
    jumps = [abs(c["loudness_jump_lu"]) for c in cuts if c["loudness_jump_lu"] is not None]
    warnings = []
    if soundtrack.duration_s + soundtrack.offset_s < float(shots[-1].end) - 1:
        warnings.append("音轨比画面短，结尾镜头没有声音数据")
    if abs(soundtrack.offset_s) > 0.5:
        warnings.append(f"音轨起点与首帧相差 {soundtrack.offset_s:.2f} 秒，已按容器时间对齐")
    summary = dict(
        integrated_lufs=_round(gated_loudness(np.concatenate(blocks)) if blocks else None),
        lra_lu=_round(loudness_range(np.concatenate(short)) if short else None),
        silence_share=_round(silent_s / covered, 4) if covered else None,
        cuts=len(cuts),
        cuts_in_silence_share=_round(sum(1 for c in judged if c["in_silence"]) / len(judged), 4) if judged else None,
        relation_counts=relations,
        median_abs_sound_offset_s=_round(_median(offsets), 3),
        median_abs_loudness_jump_lu=_round(_median(jumps)))
    return dict(
        shots=rows, cuts=cuts, summary=summary, curve=soundtrack.curve(float(context.duration)),
        duration_s=float(context.duration), offset_s=round(soundtrack.offset_s, 4), warnings=warnings,
        parameters=dict(
            loudness="ITU-R BS.1770-4 / EBU R 128 via ffmpeg ebur128; per-Shot integrated loudness "
                     "re-gated from 400 ms momentary blocks lying wholly inside the Shot",
            lra="EBU Tech 3342 from 3 s short-term values; only Shots of 3 s or longer",
            silence=f"mono RMS < {SILENCE_DB:g} dBFS for >= {SILENCE_MIN_S:g} s",
            cut_silence="the cut falls inside or within one hop of a silent run",
            loudness_jump="momentary loudness of the 400 ms after the cut minus the 400 ms before",
            sound_change=f"mean change of {BANDS} log-band levels between {NOVELTY_HALF_S:g} s before and after; "
                         f">= {CHANGE_DB:g} dB counts; nearest within +-{CUT_WINDOW_S:g} s of the cut",
            relation=f"sync if |offset| <= {SYNC_FRAMES} frames, J if the sound changes before the cut, "
                     "L if after; candidates to check by listening",
            sample_rate=SAMPLE_RATE, hop_s=round(HOP / SAMPLE_RATE, 5)),
        scope=context.scope)


# -- Chromaprint fingerprints and cross-episode matching -----------------------

def pack_fingerprint(fingerprint, offset_s, index):
    data = base64.b64encode(zlib.compress(np.asarray(fingerprint, dtype="<u4").tobytes(), 9)).decode("ascii")
    return dict(data=data, item_s=FP_ITEM_S, offset_s=round(float(offset_s), 4), items=int(len(fingerprint)),
                source_size=index.size, source_mtime_ns=index.mtime_ns, algorithm=1)


def unpack_fingerprint(packed):
    if not packed or not packed.get("data"):
        return None
    raw = zlib.decompress(base64.b64decode(packed["data"]))
    return np.frombuffer(raw, dtype="<u4").astype(np.uint32)


def fingerprint_only(path, ffmpeg=FFMPEG, cancel=None):
    """Chromaprint of a source without the other features (for episodes not yet analysed)."""
    with tempfile.TemporaryDirectory(prefix="cv-fp-") as folder:
        target = Path(folder) / "fp.bin"
        process = subprocess.Popen(
            [str(ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path),
             "-map", "0:a:0", "-ac", "1", "-f", "chromaprint", "-fp_format", "raw", str(target)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        while process.poll() is None:
            if cancel is not None and cancel.is_set():
                process.kill()
                process.wait()
                _cancel(cancel)
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
        error = process.stderr.read().decode("utf-8", "replace").strip()
        if process.returncode != 0 or not target.exists():
            raise RuntimeError(f"ffmpeg 计算音频指纹失败: {error or process.returncode}")
        return np.frombuffer(target.read_bytes(), dtype="<u4").astype(np.uint32)


def _popcount(values):
    return np.bitwise_count(values) if hasattr(np, "bitwise_count") else \
        np.unpackbits(values.view(np.uint8).reshape(-1, 4), axis=1).sum(axis=1)


def match_fingerprints(a, b, max_bits=FP_MAX_BITS, min_s=FP_MIN_S, max_gap_s=FP_MAX_GAP_S, candidates=8):
    """Long stretches of shared audio between two fingerprints.

    Returns [{a: [start, end), b: [start, end), seconds, mean_bit_errors}] in item units.
    """
    a, b = np.asarray(a, dtype=np.uint32), np.asarray(b, dtype=np.uint32)
    if a.size == 0 or b.size == 0:
        return []
    def keys(values):
        # 高低 16 位各投一票：重新编码造成的单个比特翻转只毁掉其中一半。
        values = values.tolist()
        return [(v >> 16, 0x10000 | (v & 0xFFFF)) for v in values]

    positions = {}
    for j, pair in enumerate(keys(b)):
        for key in pair:
            positions.setdefault(key, []).append(j)
    own = {}
    for pair in keys(a):
        for key in pair:
            own[key] = own.get(key, 0) + 1
    votes = {}
    for i, pair in enumerate(keys(a)):
        for key in pair:
            hits = positions.get(key)
            if not hits or len(hits) > FP_COMMON or own[key] > FP_COMMON:
                continue
            for j in hits:
                votes[j - i] = votes.get(j - i, 0) + 1
    shifts = [s for s, n in sorted(votes.items(), key=lambda item: -item[1]) if n >= 4][:candidates]
    min_items = int(round(min_s / FP_ITEM_S))
    gap = int(round(max_gap_s / FP_ITEM_S))
    found = []
    for shift in shifts:
        lo, hi = max(0, -shift), min(a.size, b.size - shift)
        if hi - lo < min_items:
            continue
        errors = _popcount(a[lo:hi] ^ b[lo + shift:hi + shift]).astype(np.float64)
        # Averaged over ~1 s: unrelated but similar-sounding audio (noise beds) scatters
        # single low-error items that would otherwise stretch a match through the gap rule.
        kernel = np.ones(FP_SMOOTH)
        smooth = np.convolve(errors, kernel, mode="same") / np.convolve(np.ones_like(errors), kernel, mode="same")
        good = np.flatnonzero(smooth <= max_bits)
        if good.size == 0:
            continue
        runs, first, last = [], good[0], good[0]
        for k in good[1:]:
            if k - last > gap:
                runs.append((first, last + 1))
                first = k
            last = k
        runs.append((first, last + 1))
        for start, stop in runs:
            if stop - start < min_items:
                continue
            span = a[lo + start:lo + stop]
            if len(np.unique(span)) < 0.3 * span.size:
                continue  # silence or a sustained tone, not a shared cue
            found.append(dict(a=[int(lo + start), int(lo + stop)],
                              b=[int(lo + start + shift), int(lo + stop + shift)],
                              seconds=round(float((stop - start) * FP_ITEM_S), 2),
                              mean_bit_errors=round(float(errors[start:stop].mean()), 2)))
    found.sort(key=lambda m: -m["seconds"])
    kept = []
    for match in found:
        if all(match["a"][1] <= k["a"][0] or k["a"][1] <= match["a"][0] for k in kept):
            kept.append(match)
    return sorted(kept, key=lambda m: m["a"][0])


def frame_at(index, seconds):
    """First frame whose origin-relative start time is >= seconds (clamped)."""
    from bisect import bisect_left
    target = index.pts[0] + seconds / float(index.tb)
    return max(0, min(len(index.pts), bisect_left(index.pts, target)))


def episode_suggestions(index, own, offset_s, others, duration_s):
    """Merge matches against other episodes into region suggestions on this project's frames.

    `others` = [(label, fingerprint, offset_s)]. A suggestion in the first quarter is
    labelled OP, in the last quarter ED, otherwise other.
    """
    spans = []
    for label, fingerprint, other_offset in others:
        for match in match_fingerprints(own, fingerprint):
            start = match["a"][0] * FP_ITEM_S + offset_s
            end = match["a"][1] * FP_ITEM_S + offset_s
            spans.append(dict(start_s=start, end_s=end, source=label, seconds=match["seconds"],
                              other_start_s=round(match["b"][0] * FP_ITEM_S + other_offset, 2),
                              mean_bit_errors=match["mean_bit_errors"]))
    spans.sort(key=lambda s: s["start_s"])
    merged = []
    for span in spans:
        if merged and span["start_s"] < merged[-1]["end_s"]:
            last = merged[-1]
            last["end_s"] = max(last["end_s"], span["end_s"])
            last["matches"].append(span)
        else:
            merged.append(dict(start_s=span["start_s"], end_s=span["end_s"], matches=[span]))
    result = []
    for item in merged:
        start, end = max(0.0, item["start_s"]), min(duration_s, item["end_s"])
        if end - start < FP_MIN_S / 2:
            continue
        label = "OP" if start < duration_s / 4 else ("ED" if end > duration_s * 3 / 4 else "other")
        result.append(dict(label=label, start_s=round(start, 2), end_s=round(end, 2),
                           start_frame=frame_at(index, start), end_frame=frame_at(index, end),
                           episodes=sorted({m["source"] for m in item["matches"]}),
                           matches=item["matches"]))
    return result
