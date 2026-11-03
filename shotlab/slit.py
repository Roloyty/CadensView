"""Per-Shot slit scans and evenly spaced stills (freder/cinemetrics 02_3 / 02_5)."""

from concurrent.futures import CancelledError
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from shotlab.fonts import ui_font
from shotlab.statistics import AnalysisScope, build_context
from t0.media import FrameReader

CHUNK = 24


def _cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise CancelledError("Slit-scan export cancelled; no project data was changed")


def render_shot(index, shot, width=160, cancel=None, progress=None):
    """One vertical strip per sampled source column, spanning the whole Shot."""
    width = max(8, min(int(width), 2000))
    start, end = shot.start_frame, shot.end_frame_exclusive
    span = max(1, end - start)
    output = None
    last_column = -1
    last_strip = None
    reader = FrameReader(index, sequential=True)
    done = 0
    try:
        position = start
        while position < end:
            _cancel(cancel)
            block_end = min(position + CHUNK, end)
            for offset, frame in enumerate(reader.read_range(position, block_end)):
                number = position + offset
                rgb = frame.to_ndarray(format="rgb24")
                height, source_width = rgb.shape[:2]
                if output is None:
                    output = np.zeros((height, width, 3), np.uint8)
                center = source_width // 2
                strip = rgb[:, max(0, center - 1):min(source_width, center + 2)].mean(axis=1)
                strip = np.clip(strip, 0, 255).astype(np.uint8)
                column = min(width - 1, int((number - start) * width / span))
                if column > last_column:
                    output[:, last_column + 1:column + 1] = strip[:, None, :]
                    last_column = column
                last_strip = strip
                done += 1
            position = block_end
            if progress is not None:
                progress(done, span)
    finally:
        reader.close()
    if output is None:
        raise ValueError("镜头没有可解码帧")
    if last_column < width - 1 and last_strip is not None:
        output[:, last_column + 1:] = last_strip[:, None, :]
    return output


def build_sheet(index, shots, path, width=160, columns=6, cell_height=120,
                cancel=None, progress=None):
    if not shots:
        raise ValueError("没有可导出的已完成镜头")
    total = sum(shot.end_frame_exclusive - shot.start_frame for shot in shots)
    done = 0
    cells = []
    for shot in shots:
        _cancel(cancel)
        strip = render_shot(index, shot, width=width, cancel=cancel,
                            progress=lambda part, span: (progress(done + part, total) if progress else None))
        image = Image.fromarray(strip, "RGB").resize(
            (max(8, int(cell_height * 0.55)), cell_height), Image.Resampling.LANCZOS)
        cells.append((image, shot))
        done += shot.end_frame_exclusive - shot.start_frame
    label_height = 22
    gap = 8
    columns = max(1, min(int(columns), 12))
    rows = (len(cells) + columns - 1) // columns
    cell_width = max(image.width for image, _ in cells)
    sheet = Image.new("RGB", (columns * (cell_width + gap) + gap,
                              rows * (cell_height + label_height + gap) + gap), (22, 22, 26))
    draw = ImageDraw.Draw(sheet)
    font = ui_font(11)
    for i, (image, shot) in enumerate(cells):
        row, column = divmod(i, columns)
        x = gap + column * (cell_width + gap) + (cell_width - image.width) // 2
        y = gap + row * (cell_height + label_height + gap)
        sheet.paste(image, (x, y))
        label = f"#{shot.shot_number}  {float(shot.duration):.2f}s"
        draw.text((gap + column * (cell_width + gap), y + cell_height + 3),
                  label, fill=(205, 205, 212), font=font)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, "PNG")
    return {"path": str(target.resolve()), "written": len(cells), "columns": columns}


def export_stills(project, index, path, count=100, scope=AnalysisScope(),
                  width=480, cancel=None, progress=None):
    """Evenly spaced stills across the selected completed Shots."""
    context = build_context(project, index, scope)
    shots = context.records
    if not shots:
        raise ValueError("没有可导出的已完成镜头")
    count = max(1, min(int(count), 2000))
    total_frames = sum(shot.end_frame_exclusive - shot.start_frame for shot in shots)
    if total_frames < 1:
        raise ValueError("范围内没有帧")
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    targets = []
    for i in range(count):
        offset = int((i + 0.5) * total_frames / count)
        remaining = offset
        for shot in shots:
            span = shot.end_frame_exclusive - shot.start_frame
            if remaining < span:
                targets.append(shot.start_frame + remaining)
                break
            remaining -= span
        else:
            targets.append(shots[-1].end_frame_exclusive - 1)
    reader = FrameReader(index, sequential=True)
    written = 0
    try:
        for done, number in enumerate(targets, 1):
            _cancel(cancel)
            frame = reader.read(number)
            rgb = frame.to_ndarray(format="rgb24")
            height, source_width = rgb.shape[:2]
            thumb_width = max(32, min(int(width), source_width))
            thumb_height = max(1, round(height * thumb_width / source_width))
            image = Image.fromarray(rgb, "RGB").resize(
                (thumb_width, thumb_height), Image.Resampling.LANCZOS)
            image.save(folder / f"still_{done:04d}.jpg", "JPEG", quality=90)
            written += 1
            if progress is not None:
                progress(done, count)
    finally:
        reader.close()
    return {"path": str(folder.resolve()), "written": written, "count": count}
