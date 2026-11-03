"""Frame-difference motion measurement (adapted from freder/cinemetrics 04_1).

Motion is the share of channel samples whose absolute difference from the
previous frame exceeds a fixed threshold. The first frame of each Shot has no
in-Shot predecessor, so it is excluded from the Shot average rather than
counting the hard cut as motion. The same pass classifies each frame pair for
animation timing (held drawings, camera moves, drawn motion; see animation.py).
"""

from collections import Counter
from concurrent.futures import CancelledError
from pathlib import Path

from PIL import Image, ImageDraw

from shotlab.animation import (HOLD_MEAN, HOLD_SHARE, KOMA_LABELS, PAN_MIN_PX, THRESHOLD,
                               PairClassifier, rate_warning, summarize_shot)
from shotlab.fonts import ui_font
from shotlab.statistics import AnalysisScope, build_context
from t0.media import FrameReader

CHUNK = 24
MAX_CURVE_POINTS = 3000
ANALYSIS_WIDTH = 320  # 运动与拍数在缩小画面上算：抗压缩噪声，也让长片可行


def _cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise CancelledError("Motion analysis cancelled; project data was not changed")


def _mean(values):
    return float(sum(values) / len(values)) if values else None


def analysis_frame(frame, width=ANALYSIS_WIDTH):
    """RGB array no wider than `width`; small sources are used as decoded."""
    if frame.width > width:
        height = max(2, round(frame.height * width / frame.width / 2) * 2)
        return frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
    return frame.to_ndarray(format="rgb24")


def analyze_motion(project, index, scope=AnalysisScope(), cancel=None, progress=None):
    """Decode the selected completed Shots; measure motion, holds (koma) and camera moves."""
    context = build_context(project, index, scope)
    shots = context.records
    if not shots:
        raise ValueError("没有可分析的已完成镜头")
    total = sum(shot.end_frame_exclusive - shot.start_frame for shot in shots)
    done = 0
    curve = []  # (frame, motion)
    rows = []
    classifier = PairClassifier()
    reader = FrameReader(index, sequential=True)
    try:
        for shot in shots:
            values, kinds, shifts, locals_ = [], [], [], []
            previous = None
            position = shot.start_frame
            while position < shot.end_frame_exclusive:
                _cancel(cancel)
                end = min(position + CHUNK, shot.end_frame_exclusive)
                frames = reader.read_range(position, end)
                for offset, frame in enumerate(frames):
                    number = position + offset
                    rgb = analysis_frame(frame)
                    if previous is not None and previous.shape == rgb.shape:
                        pair = classifier.classify(previous, rgb)
                        values.append(pair["motion"])
                        curve.append((number, pair["motion"]))
                        kinds.append(pair["kind"]); shifts.append(pair["shift"]); locals_.append(pair["local"])
                    previous = rgb
                    done += 1
                position = end
                if progress is not None:
                    progress(done, total)
            fps = (shot.end_frame_exclusive - shot.start_frame) / float(shot.duration) if shot.duration else None
            timing = summarize_shot(shot.start_frame, kinds, shifts, locals_, fps,
                                    previous.shape[1] if previous is not None else None)
            rows.append(dict(shot.to_dict(), motion=_mean(values), motion_samples=len(values), **timing))
    finally:
        reader.close()
    _cancel(cancel)
    stride = max(1, len(curve) // MAX_CURVE_POINTS)
    binned = []
    for start in range(0, len(curve), stride):
        part = curve[start:start + stride]
        if not part:
            continue
        binned.append(dict(
            start_frame=part[0][0], end_frame=part[-1][0] + 1,
            video_time_s=float((index.pts[part[0][0]] - index.pts[0]) * index.tb),
            motion=round(_mean([value for _, value in part]), 6)))
    timed = [row for row in rows if row.get("koma")]
    koma_counts = Counter(row["koma"] for row in timed if row["koma_reliable"])
    warnings = [w for w in [rate_warning(index.metadata.get("average_rate"))] if w]
    return dict(
        samples=binned, shots=rows, duration_s=float(context.duration),
        timing_summary=dict(
            shots_with_reliable_koma=sum(koma_counts.values()),
            koma_shot_counts={KOMA_LABELS[k]: koma_counts[k] for k in sorted(koma_counts)},
            mean_drawings_per_s=_mean([r["drawings_per_s"] for r in timed if r.get("drawings_per_s")]),
            mean_still_share=_mean([r["still_share"] for r in timed if r.get("still_share") is not None]),
            mean_pan_share=_mean([r["pan_share"] for r in timed if r.get("pan_share") is not None]),
            koma_changing_shots=sum(1 for r in timed if r.get("koma_segments"))),
        warnings=warnings,
        parameters=dict(threshold=THRESHOLD, first_frame="excluded",
                        definition="share of RGB channel samples with |delta| > threshold",
                        analysis_width=ANALYSIS_WIDTH, hold_share=HOLD_SHARE, hold_mean=HOLD_MEAN,
                        pan_min_px=PAN_MIN_PX, koma="mode of inner exposure lengths (1/2/3/4+)",
                        curve_points=len(binned)),
        scope=context.scope)


def write_spectrum(result, path, width=1800, height=300, accent=(232, 163, 61)):
    """Render a motion fingerprint: source-time x motion area chart."""
    samples = result.get("samples") or []
    duration = max(float(result.get("duration_s") or 0), 1e-6)
    width = max(320, min(int(width), 4000))
    height = max(120, min(int(height), 1000))
    image = Image.new("RGB", (width, height), (18, 18, 22))
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 52, width - 16, 18, height - 34
    for i in range(5):
        y = bottom - (bottom - top) * i / 4
        draw.line([(left, y), (right, y)], fill=(48, 48, 56))
        draw.text((8, y - 7), f"{i / 4:.2f}", fill=(150, 150, 158), font=ui_font(11))
    for i in range(5):
        x = left + (right - left) * i / 4
        seconds = duration * i / 4
        draw.line([(x, top), (x, bottom)], fill=(38, 38, 45))
        draw.text((x - 18, bottom + 8), f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}",
                  fill=(150, 150, 158), font=ui_font(11))
    for shot in (result.get("shots") or [])[1:]:
        x = left + (right - left) * min(1.0, max(0.0, shot["start_time_s"] / duration))
        draw.line([(x, top), (x, bottom)], fill=(70, 70, 80))
    if samples:
        points = []
        for sample in samples:
            x = left + (right - left) * min(1.0, sample["video_time_s"] / duration)
            y = bottom - (bottom - top) * min(1.0, max(0.0, sample["motion"]))
            points.append((x, y))
        if len(points) == 1:
            points.append((points[0][0] + 1, points[0][1]))
        draw.polygon([(left, bottom), *points, (right, bottom)],
                     fill=(accent[0] // 4, accent[1] // 4, accent[2] // 4))
        draw.line(points, fill=accent, width=2)
    draw.text((left, 2), "motion  |  threshold 10/255  |  first frame of Shot excluded",
              fill=(190, 190, 198), font=ui_font(12))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, "PNG")
    return {"path": str(target.resolve()), "width": width, "height": height}


def render_spectrum(result, path, **options):
    return write_spectrum(result, path, **options)
