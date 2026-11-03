"""Repeated footage (bank/流用 shots) by perceptual hash of three frames per Shot.

Each completed Shot is sampled at 25/50/75% of its length; a 64-bit dHash of a
9x8 grey thumbnail is kept per sample. Two Shots match when their middle samples
are close and one more sample agrees, so a re-used shot trimmed differently
still matches. Near-flat frames (black, white, title cards) carry no hash because
any two of them would match.
"""

from concurrent.futures import CancelledError

import numpy as np

from shotlab.statistics import AnalysisScope, build_context
from t0.media import FrameReader

SAMPLES = (0.25, 0.5, 0.75)
MATCH_BITS = 6     # Hamming distance for the middle sample
SUPPORT_BITS = 8   # ...and for at least one other sample
FLAT_STD = 6.0


def dhash(rgb):
    """64-bit difference hash of an RGB array, or None for a near-flat frame."""
    from PIL import Image
    grey = Image.fromarray(rgb, "RGB").convert("L")
    small = np.asarray(grey.resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    if float(np.asarray(grey.resize((32, 32)), dtype=np.float32).std()) < FLAT_STD:
        return None
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def shot_hashes(project, index, scope=AnalysisScope(), cancel=None, progress=None):
    """{shot_id: {start, end, hashes: [h|None]*3}} for the completed Shots in scope."""
    records = build_context(project, index, scope).records
    result = {}
    reader = FrameReader(index, sequential=False)
    try:
        for done, shot in enumerate(records, 1):
            if cancel is not None and cancel.is_set():
                raise CancelledError("重复镜头检测已取消")
            span = shot.end_frame_exclusive - shot.start_frame
            frames = sorted({shot.start_frame + min(span - 1, int(span * f)) for f in SAMPLES})
            while len(frames) < len(SAMPLES):
                frames.append(frames[-1])
            hashes = [dhash(reader.read(number).to_ndarray(format="rgb24")) for number in frames]
            result[shot.shot_id] = dict(start=shot.start_frame, end=shot.end_frame_exclusive, hashes=hashes,
                                        number=shot.shot_number, duration_s=float(shot.duration))
            if progress is not None:
                progress(done, len(records))
    finally:
        reader.close()
    return result


def is_match(first, second):
    a, b = first["hashes"], second["hashes"]
    if a[1] is None or b[1] is None or hamming(a[1], b[1]) > MATCH_BITS:
        return False
    return any(x is not None and y is not None and hamming(x, y) <= SUPPORT_BITS
               for i, x in enumerate(a) for j, y in enumerate(b) if (i, j) != (1, 1))


def groups(entries, adjacent_ok=False):
    """Union-find over matching Shots; entries = [(key, entry)] with entry['number'] ordering."""
    parent = {key: key for key, _ in entries}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    ordered = sorted(entries, key=lambda item: (item[1].get("source", ""), item[1]["number"]))
    for i, (key, entry) in enumerate(ordered):
        for other_key, other in ordered[i + 1:]:
            same_source = entry.get("source", "") == other.get("source", "")
            # 同片相邻镜头相似多半是一个镜头被误拆，不算流用。
            if same_source and not adjacent_ok and abs(entry["number"] - other["number"]) <= 1:
                continue
            if is_match(entry, other):
                parent[find(other_key)] = find(key)
    clusters = {}
    for key, _ in entries:
        clusters.setdefault(find(key), []).append(key)
    by_key = dict(entries)
    return [sorted(members, key=lambda k: (by_key[k].get("source", ""), by_key[k]["number"]))
            for members in clusters.values() if len(members) > 1]


def find_repeats(hashes):
    """Groups of repeated Shots inside one project."""
    entries = list(hashes.items())
    result = []
    for members in groups(entries):
        rows = [dict(shot_id=key, shot_number=hashes[key]["number"], start_frame=hashes[key]["start"],
                     duration_s=hashes[key]["duration_s"]) for key in members]
        result.append(dict(count=len(rows), shots=rows))
    return sorted(result, key=lambda g: (-g["count"], g["shots"][0]["shot_number"]))


def shared_sequences(first, second, min_run=6):
    """Runs of consecutive matching Shots between two projects (e.g. OP/ED, recaps).

    `first`/`second` are hash dicts. Returns [{first: [start,end), second: [start,end), shots}].
    """
    a = sorted(first.values(), key=lambda e: e["number"])
    b = sorted(second.values(), key=lambda e: e["number"])
    matches = {(i, j) for i, x in enumerate(a) for j, y in enumerate(b) if is_match(x, y)}
    runs, used = [], set()
    for i, j in sorted(matches):
        if (i, j) in used:
            continue
        length = 0
        while (i + length, j + length) in matches:
            used.add((i + length, j + length))
            length += 1
        if length >= min_run:
            runs.append(dict(shots=length,
                             first=[a[i]["start"], a[i + length - 1]["end"]],
                             second=[b[j]["start"], b[j + length - 1]["end"]],
                             first_shot_numbers=[a[i]["number"], a[i + length - 1]["number"]],
                             second_shot_numbers=[b[j]["number"], b[j + length - 1]["number"]]))
    return sorted(runs, key=lambda r: -r["shots"])
