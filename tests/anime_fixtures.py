"""Synthetic animation clips: drawings held on 1s/2s/3s, and a camera pan over a still layer."""

from fractions import Fraction

import av
import numpy as np


def _write(path, frames, rate=24):
    with av.open(str(path), "w") as output:
        stream = output.add_stream("ffv1", rate=rate)
        stream.width, stream.height, stream.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv444p"
        for number, array in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts = number
            frame.time_base = Fraction(1, rate)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return path


def held_square(koma, count=48, size=(72, 128)):
    """A square that moves to a new drawing every `koma` frames."""
    height, width = size
    frames = []
    for number in range(count):
        drawing = number // koma
        array = np.full((height, width, 3), 235, np.uint8)
        x = 8 + (drawing * 7) % (width - 30)
        array[20:44, x:x + 22] = [30, 60, 200]
        frames.append(array)
    return frames


def panning_layer(count=48, size=(72, 128), step=3):
    """A textured background slid sideways every frame; nothing is redrawn."""
    height, width = size
    rng = np.random.default_rng(7)
    texture = rng.integers(0, 255, (height, width * 4, 3), dtype=np.uint8)
    texture = np.repeat(np.repeat(texture[::4, ::4], 4, axis=0), 4, axis=1)[:height]
    return [np.ascontiguousarray(texture[:, number * step:number * step + width]) for number in range(count)]


def anime_clip(path):
    """Shots: on 3s [0,36), on 2s [36,72), on 1s [72,96), pan [96,144)."""
    frames = held_square(3, 36) + held_square(2, 36) + held_square(1, 24) + panning_layer(48)
    return _write(path, frames)
