"""Compare several .shotlab projects: robust summaries, dominance tests, episode trends.

Films and episodes are compared through their shot-length distributions, not
their means: medians with order-statistic intervals, cutting rates with Poisson
intervals, Mann-Whitney U and Cliff's delta for each pair (Redfern 2014). Shared
shot sequences between projects come from stored perceptual hashes (repeats.run)
and usually mark an opening/ending or a recap.
"""

from itertools import combinations
from pathlib import Path

from shotlab.repeats import shared_sequences
from shotlab.statistics.robust import cliffs_delta, mann_whitney, summary
from shotlab.storage import ProjectStore


def _overlaps(start, end, regions):
    return any(start < r["end_frame"] and r["start_frame"] < end for r in regions)


def load_project(path, exclude_regions=True):
    store = ProjectStore(path)
    try:
        project, index, settings = store.load()
    finally:
        store.close()
    regions = settings.get("regions", []) if exclude_regions else []
    tb = index.tb
    end_pts = index.pts[-1] + (index.durations[-1] or 0)
    durations, excluded = [], 0
    for row in range(project.completed_count):
        first, end = project.interval(row)
        if _overlaps(first, end, regions):
            excluded += 1
            continue
        stop = index.pts[end] if end < len(index.pts) else end_pts
        durations.append(float((stop - index.pts[first]) * tb))
    metadata = settings.get("metadata", {})
    label = metadata.get("title") or Path(path).stem
    if metadata.get("episode"):
        label = f"{label} #{metadata['episode']}"
    return dict(path=str(Path(path).resolve()), label=label, metadata=metadata, durations=durations,
                excluded_shots=excluded, complete=project.pending_start is None,
                hashes=settings.get("shot_hashes", {}))


def _episode_key(film):
    value = film["metadata"].get("episode", "")
    try:
        return (0, float(value), film["label"])
    except (TypeError, ValueError):
        return (1, 0, str(value) or film["label"])


def compare_projects(paths, exclude_regions=True, group_by=None):
    if len(paths) < 2:
        raise ValueError("至少选择两个项目")
    films = [load_project(path, exclude_regions) for path in paths]
    rows = []
    for film in films:
        rows.append(dict(label=film["label"], path=film["path"], metadata=film["metadata"],
                         excluded_shots=film["excluded_shots"], complete=film["complete"],
                         **summary(film["durations"])))
    pairs = []
    for a, b in combinations(range(len(films)), 2):
        first, second = films[a]["durations"], films[b]["durations"]
        test, dominance = mann_whitney(first, second), cliffs_delta(first, second)
        pairs.append(dict(first=films[a]["label"], second=films[b]["label"],
                          mann_whitney=test, cliffs_delta=dominance))
    # Holm 校正：多对比较时控制族错误率。
    ordered = sorted((p for p in pairs if p["mann_whitney"]), key=lambda p: p["mann_whitney"]["p_two_sided"])
    running = 0.0
    for rank, pair in enumerate(ordered):
        adjusted = min(1.0, pair["mann_whitney"]["p_two_sided"] * (len(ordered) - rank))
        running = max(running, adjusted)
        pair["mann_whitney"]["p_holm"] = running
    groups = {}
    if group_by:
        buckets = {}
        for film in films:
            buckets.setdefault(str(film["metadata"].get(group_by) or "（未填）"), []).extend(film["durations"])
        groups = {key: summary(values) for key, values in sorted(buckets.items())}
    trend = []
    if any(f["metadata"].get("episode") for f in films):
        for film in sorted(films, key=_episode_key):
            s = summary(film["durations"])
            trend.append(dict(label=film["label"], episode=film["metadata"].get("episode", ""),
                              msl_s=s.get("msl_s"), cutting_rate_per_min=s.get("cutting_rate_per_min"),
                              shot_count=s.get("shot_count")))
    shared = []
    hashed = [f for f in films if f["hashes"]]
    for a, b in combinations(hashed, 2):
        for run in shared_sequences(a["hashes"], b["hashes"]):
            shared.append(dict(run, first_label=a["label"], second_label=b["label"]))
    return dict(films=rows, pairs=pairs, groups=groups, group_by=group_by, episode_trend=trend,
                shared_sequences=shared, exclude_regions=exclude_regions,
                hashed_projects=len(hashed),
                notes=["时长单位秒；MSL 区间为次序统计量 95% 区间，剪辑率为每分钟镜头数（Poisson 95%）",
                       "Cliff's delta > 0 表示前者的镜头普遍更长",
                       "共享片段需要各项目先运行“重复镜头”检测（保存哈希）"])


def write_compare_csv(result, path):
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["label", "shot_count", "duration_s", "msl_s", "msl_ci95_s", "asl_s", "q1_s", "q3_s", "iqr_s",
              "cutting_rate_per_min", "cutting_rate_ci95_per_min", "excluded_shots", "complete", "path"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in result["films"]:
            writer.writerow({k: row.get(k) for k in fields})
    return str(path.resolve())
