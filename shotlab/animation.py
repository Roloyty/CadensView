"""Animation timing from frame pairs: held drawings (koma), camera moves and drawn motion.

Limited animation holds one drawing for several frames ("on twos/threes"). A pair
of frames counts as a *hold* when almost no pixel changes. When the whole image
shifts, phase correlation estimates the global (camera/cel slide) displacement;
after compensating it, the residual change is the drawn (sakuga) motion. A new
drawing is a pair that is neither a hold nor a pure global shift.

Known limits, reported per Shot instead of hidden: follow-pans that keep a
character fixed over a moving background look like a new drawing every frame,
so koma is flagged unreliable when most pairs are pans; telecined 29.97/59.94
sources repeat fields in a 3:2 pattern and need inverse telecine first.
"""

from collections import Counter
from fractions import Fraction

import numpy as np

HOLD_SHARE = 0.002   # share of samples allowed to change in a held pair (compression noise)
HOLD_MEAN = 1.0      # mean absolute difference (0-255) allowed in a held pair
PAN_MIN_PX = 0.5     # global shift that counts as a camera move, in analysis pixels
PAN_PEAK = 0.05      # minimum normalised phase-correlation peak for a trusted shift
PAN_MIN_COVERAGE = 0.2  # a pan changes a large share of the frame before compensation
PAN_EXPLAINED = 0.5     # ...and compensation must remove at least half of that change
THRESHOLD = 10
KOMA_LABELS = {1: "一拍一", 2: "一拍二", 3: "一拍三", 4: "一拍四及以上"}
UNRELIABLE_PAN_SHARE = 0.3


def _window(shape):
    return np.outer(np.hanning(shape[0]), np.hanning(shape[1])).astype(np.float32)


def phase_shift(previous, current, window=None):
    """Integer (dy, dx) that best aligns previous onto current, and its peak height."""
    window = _window(previous.shape) if window is None else window
    a = np.fft.rfft2(previous * window)
    b = np.fft.rfft2(current * window)
    cross = b * np.conj(a)
    cross /= np.abs(cross) + 1e-9
    correlation = np.fft.irfft2(cross, s=previous.shape)
    y, x = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    height, width = previous.shape
    dy = y - height if y > height // 2 else y
    dx = x - width if x > width // 2 else x
    return int(dy), int(dx), float(correlation[y, x])


def _changed_share(first, second):
    difference = np.abs(first.astype(np.int16) - second.astype(np.int16))
    return float(np.count_nonzero(difference > THRESHOLD)) / float(difference.size), float(difference.mean())


def _residual(previous, current, dy, dx):
    """Changed share after shifting previous by (dy, dx), on the overlap only."""
    height, width = previous.shape[:2]
    if abs(dy) >= height // 2 or abs(dx) >= width // 2:
        return 1.0
    ys = slice(max(0, dy), height + min(0, dy))
    xs = slice(max(0, dx), width + min(0, dx))
    ps = slice(max(0, -dy), height + min(0, -dy))
    qs = slice(max(0, -dx), width + min(0, -dx))
    return _changed_share(previous[ps, qs], current[ys, xs])[0]


class PairClassifier:
    """Classify consecutive analysis frames of one source; keeps an FFT window per size."""

    def __init__(self):
        self.windows = {}

    def classify(self, previous, current):
        motion, mean = _changed_share(previous, current)
        if motion < HOLD_SHARE and mean < HOLD_MEAN:
            return dict(motion=motion, kind="hold", shift=0.0, local=0.0)
        gray_prev = previous.mean(axis=2, dtype=np.float32)
        gray_cur = current.mean(axis=2, dtype=np.float32)
        window = self.windows.get(gray_prev.shape)
        if window is None:
            window = self.windows[gray_prev.shape] = _window(gray_prev.shape)
        dy, dx, peak = phase_shift(gray_prev, gray_cur, window)
        shift = float(np.hypot(dy, dx))
        # 纯色背景上的单个物体移动也会给出平移峰值；只有平移解释了大面积变化才算镜头/赛璐璐移动。
        if shift >= PAN_MIN_PX and peak >= PAN_PEAK and motion >= PAN_MIN_COVERAGE:
            local = _residual(previous, current, dy, dx)
            if local <= motion * PAN_EXPLAINED:
                kind = "pan_hold" if local < HOLD_SHARE * 5 else "pan_draw"
                return dict(motion=motion, kind=kind, shift=shift, local=local)
        return dict(motion=motion, kind="draw", shift=0.0, local=motion)


def koma_bucket(length):
    return min(int(length), 4)


def summarize_shot(start_frame, kinds, shifts, locals_, fps, width):
    """Per-Shot timing summary from the pair classifications (pairs start at start_frame+1)."""
    pairs = len(kinds)
    if not pairs:
        return dict(pairs=0, koma=None, koma_label=None, koma_reliable=False, exposures={},
                    drawings=1, drawings_per_s=None, still_share=None, pan_share=None,
                    pan_speed=None, drawn_motion=None, koma_segments=[])
    exposures = []  # (first_frame, length)
    first, length = start_frame, 1
    for offset, kind in enumerate(kinds, 1):
        if kind in ("draw", "pan_draw"):
            exposures.append((first, length))
            first, length = start_frame + offset, 1
        else:
            length += 1
    exposures.append((first, length))
    # 首尾曝光被切点截断，长度不代表原拍数；内部曝光够多时只用内部。
    inner = exposures[1:-1] if len(exposures) >= 4 else exposures
    counts = Counter(koma_bucket(n) for _, n in inner)
    koma = max(sorted(counts), key=lambda k: counts[k]) if counts else None
    pan_pairs = sum(kind.startswith("pan") for kind in kinds)
    pan_share = pan_pairs / pairs
    moved = [v for v, kind in zip(locals_, kinds) if kind != "hold"]
    segments = []
    for first_frame, n in exposures:
        bucket = koma_bucket(n)
        if segments and segments[-1]["koma"] == bucket:
            segments[-1].update(end_frame=first_frame + n, exposures=segments[-1]["exposures"] + 1)
        else:
            segments.append(dict(start_frame=first_frame, end_frame=first_frame + n, koma=bucket, exposures=1))
    # 少于 3 张画的片段视为抖动，并入前段，避免把单次停顿读成变拍。
    merged = []
    for segment in segments:
        if merged and (segment["exposures"] < 3 or merged[-1]["koma"] == segment["koma"]):
            merged[-1].update(end_frame=segment["end_frame"], exposures=merged[-1]["exposures"] + segment["exposures"])
        else:
            merged.append(dict(segment))
    duration = (pairs + 1) / fps if fps else None
    drawings = len(exposures)
    return dict(
        pairs=pairs, koma=koma, koma_label=KOMA_LABELS.get(koma),
        koma_reliable=bool(koma) and pan_share < UNRELIABLE_PAN_SHARE and len(inner) >= 3,
        exposures={str(k): counts[k] for k in sorted(counts)},
        drawings=drawings, drawings_per_s=drawings / duration if duration else None,
        still_share=sum(kind == "hold" for kind in kinds) / pairs,
        pan_share=pan_share,
        pan_speed=(float(np.mean([s for s in shifts if s])) / width * fps) if pan_pairs and width and fps else 0.0,
        drawn_motion=float(np.mean(moved)) if moved else 0.0,
        koma_segments=[dict(s, koma_label=KOMA_LABELS[s["koma"]]) for s in merged] if len(merged) > 1 else [])


def rate_warning(average_rate):
    """Telecine/PAL notes for koma counting, from the stream's average rate."""
    try:
        rate = float(Fraction(str(average_rate)))  # "24000/1001" 之类的有理数
    except (ValueError, ZeroDivisionError):  # 元数据缺失只是不给提示
        return None
    if abs(rate - 29.97) < 0.05 or abs(rate - 59.94) < 0.1 or abs(rate - 30) < 0.05 or abs(rate - 60) < 0.1:
        return "源为 30/60 帧系：动画多为 24 帧经 3:2 下拉，拍数需先做反胶片过带（IVTC）再统计"
    if abs(rate - 25) < 0.05 or abs(rate - 50) < 0.1:
        return "源为 25/50 帧（PAL 加速或转制），拍数按源帧计，与 24 帧原画时间表不完全对应"
    return None
