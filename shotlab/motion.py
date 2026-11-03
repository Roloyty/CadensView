"""Frame-difference motion measurement (adapted from freder/cinemetrics 04_1).

Motion is the share of channel samples whose absolute difference from the
previous frame exceeds a fixed threshold. The first frame of each Shot has no
in-Shot predecessor, so it is excluded from the Shot average rather than
counting the hard cut as motion.
"""

from concurrent.futures import CancelledError
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from shotlab.fonts import ui_font
from shotlab.statistics import AnalysisScope, build_context
from t0.media import FrameReader

THRESHOLD = 10
CHUNK = 24
MAX_CURVE_POINTS = 3000


def _cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise CancelledError("Motion analysis cancelled; project data was not changed")


def _mean(values):
    return float(sum(values) / len(values)) if values else None


def analyze_motion(project, index, scope=AnalysisScope(), cancel=None, progress=None):
    """Decode the selected completed Shots and measure per-frame motion."""
    context = build_context(project, index, scope)
    shots = context.records
    if not shots:
        raise ValueError("没有可分析的已完成镜头")
    total = sum(shot.end_frame_exclusive - shot.start_frame for shot in shots)
    done = 0
    curve = []  # (frame, motion)
    rows = []
    reader = FrameReader(index, sequential=True)
    try:
        for shot in shots:
            values = []
            previous = None
            position = shot.start_frame
            while position < shot.end_frame_exclusive:
                _cancel(cancel)
                end = min(position + CHUNK, shot.end_frame_exclusive)
                frames = reader.read_range(position, end)
                for offset, frame in enumerate(frames):
                    number = position + offset
                    rgb = frame.to_ndarray(format="rgb24")
                    if previous is not None and previous.shape == rgb.shape:
                        difference = np.abs(rgb.astype(np.int16) - previous.astype(np.int16))
                        value = float(np.count_nonzero(difference > THRESHOLD)) / float(rgb.size)
                        values.append(value)
                        curve.append((number, value))
                    previous = rgb
                    done += 1
                position = end
                if progress is not None:
                    progress(done, total)
            rows.append(dict(shot.to_dict(), motion=_mean(values), motion_samples=len(values)))
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
    return dict(
        samples=binned, shots=rows, duration_s=float(context.duration),
        parameters=dict(threshold=THRESHOLD, first_frame="excluded",
                        definition="share of RGB channel samples with |delta| > threshold",
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
