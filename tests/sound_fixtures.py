"""Synthetic videos with soundtracks, built by the bundled ffmpeg from lavfi sources."""

import subprocess
from pathlib import Path

FFMPEG = Path(__file__).resolve().parent.parent / "tools" / "ffmpeg.exe"


def _mux(path, seconds, audio_parts, rate=24):
    """A flat grey picture of `seconds` with the mono `audio_parts` (lavfi sources) concatenated."""
    inputs = ["-f", "lavfi", "-i", f"color=gray:s=64x48:r={rate}:d={seconds}"]
    for part in audio_parts:
        inputs += ["-f", "lavfi", "-i", part]
    labels = "".join(f"[{n}:a]" for n in range(1, len(audio_parts) + 1))
    graph = f"{labels}concat=n={len(audio_parts)}:v=0:a=1,aformat=sample_rates=48000:channel_layouts=stereo[a]"
    subprocess.run([str(FFMPEG), "-nostdin", "-loglevel", "error", "-y", *inputs,
                    "-filter_complex", graph, "-map", "0:v", "-map", "[a]",
                    "-c:v", "ffv1", "-c:a", "pcm_s16le", "-shortest", str(path)], check=True)
    return path


def _sine(freq, seconds):
    return f"sine=f={freq}:d={seconds}:sample_rate=48000,volume=0.3"


def _noise(seconds, colour="pink", seed=1, tremolo=None):
    source = f"anoisesrc=d={seconds}:c={colour}:seed={seed}:r=48000:a=0.3"
    return source + (f",tremolo=f={tremolo}:d=0.8" if tremolo else "")


def _silence(seconds):
    return f"anullsrc=r=48000:cl=mono:d={seconds}"


def cut_clip(path):
    """10 s at 24 fps, intended Shots [0,96) [96,168) [168,240) (cuts at 4 s and 7 s).

    Sound: tone 0-3.5 s, noise 3.5-7 s (changes 0.5 s before the first cut: a J-cut),
    silence 7-8 s (starts on the second cut), higher tone 8-10 s.
    """
    return _mux(path, 10, [_sine(440, 3.5), _noise(3.5), _silence(1), _sine(880, 2)])


def episode(path, layout):
    """A short 'episode' of 8 fps picture; layout = [(kind, seconds, seed)], kind 'cue' or 'bed'."""
    parts = [_noise(s, "pink", 1, 3) if kind == "cue" else _noise(s, "brown", seed, 2 + seed % 3)
             for kind, s, seed in layout]
    return _mux(path, sum(s for _, s, _ in layout), parts, rate=8)
