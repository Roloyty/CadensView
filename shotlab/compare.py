"""Compare several .shotlab projects: robust summaries, dominance tests, episode trends.

Films and episodes are compared through their shot-length distributions, not
their means: medians with order-statistic intervals, cutting rates with Poisson
intervals, Mann-Whitney U and Cliff's delta for each pair (Redfern 2014). Shared
shot sequences between projects come from stored perceptual hashes (repeats.run)
and usually mark an opening/ending or a recap.
"""

from itertools import combinations
from pathlib import Path
import sqlite3
from statistics import fmean

from shotlab.repeats import shared_sequences
from shotlab.statistics.robust import cliffs_delta, mann_whitney, summary
from shotlab.storage import ProjectStore
from shotlab.study_formats import manual_alignment, query_matches, stratified_summary, validate_query_clauses


def _overlaps(start, end, regions):
    return any(start < r["end_frame"] and r["start_frame"] < end for r in regions)


def load_project(path, exclude_regions=True):
    from shotlab.study import empty as empty_study
    store = ProjectStore(path)
    try:
        project, index, settings = store.load()
    finally:
        store.close()
    regions = settings.get("regions", []) if exclude_regions else []
    study = {**empty_study(), **settings.get("study", {})}
    tiers = {t.get("id"): t for t in study.get("tiers", [])}
    scenes = [e for e in study.get("events", []) if tiers.get(e.get("tier_id"), {}).get("kind") == "scene"]
    tags = settings.get("shot_tags", {})
    scales = settings.get("scale_labels", {})
    manual_scales = settings.get("scale_manual", {})
    cut_numbers = settings.get("cut_numbers", {})
    vlm_labels = settings.get("vlm_labels", {})
    tb = index.tb
    end_pts = index.pts[-1] + (index.durations[-1] or 0)
    durations, shots, excluded = [], [], 0
    metadata = settings.get("metadata", {})
    label = metadata.get("title") or Path(path).stem
    if metadata.get("episode"):
        label = f"{label} #{metadata['episode']}"
    for row in range(project.completed_count):
        first, end = project.interval(row)
        shot = project.shots[row]
        stop = index.pts[end] if end < len(index.pts) else end_pts
        duration_s = float((stop - index.pts[first]) * tb)
        scale = scales.get(shot.id)
        if not scale or scale.get("start") != first or scale.get("end") != end:
            scale = None
        vlm = vlm_labels.get(shot.id)
        if not vlm or vlm.get("start") != first or vlm.get("end") != end:
            vlm = None
        scene = max((e for e in scenes if e["start_frame"] < end and first < e["end_frame"]),
                    key=lambda e: (min(end, e["end_frame"]) - max(first, e["start_frame"]), -e["start_frame"]),
                    default=None)
        excluded_here = _overlaps(first, end, regions)
        if excluded_here:
            excluded += 1
        else:
            durations.append(duration_s)
        shots.append(dict(id=shot.id, number=row, start=first, end=end, duration_s=duration_s,
                          note=shot.note, pending=False, tags=tags.get(shot.id, []), scale=scale,
                          scale_manual=manual_scales.get(shot.id, ""), cut_number=cut_numbers.get(shot.id, ""),
                          vlm=vlm, excluded=excluded_here, scene_id=scene["id"] if scene else None,
                          strata=dict(series=str(metadata.get("series") or "（未填）"),
                                      episode=str(metadata.get("episode") or "（未填）"),
                                      film=label, scene=scene["label"] if scene else "（未标注）",
                                      shot=str(row + 1))))
    return dict(path=str(Path(path).resolve()), label=label, metadata=metadata, durations=durations,
                shots=shots, study=study,
                excluded_shots=excluded, complete=project.pending_start is None,
                hashes=settings.get("shot_hashes", {}))


def _episode_key(film):
    value = film["metadata"].get("episode", "")
    try:
        return (0, float(value), film["label"])
    except (TypeError, ValueError):
        return (1, 0, str(value) or film["label"])


def _inference_samples(film, unit):
    if unit == "shot":
        return film["durations"]
    if unit == "scene":
        scenes = {}
        for shot in film["shots"]:
            if shot["scene_id"] is not None:
                scenes.setdefault(shot["scene_id"], []).append(shot["duration_s"])
        return [fmean(values) for values in scenes.values()]
    return [fmean(film["durations"])] if film["durations"] else []


def compare_projects(paths, exclude_regions=True, group_by=None, inference_unit="film", alignment=None,
                     query=None, saved_query_id=None):
    if inference_unit not in ("shot", "scene", "film", "episode"):
        raise ValueError("推断单位须为 shot / scene / film / episode")
    if len(paths) < 2:
        raise ValueError("至少选择两个项目")
    films = []
    for path in paths:
        try:
            films.append(load_project(path, exclude_regions))
        except (sqlite3.Error, OSError, LookupError, TypeError, ValueError) as error:
            # Name the file: with several projects selected, a bare "file is not a
            # database" gives no clue which one to fix.
            raise ValueError(f"无法读取项目 {path}（{error}）") from error
    if saved_query_id:
        saved = next((q for film in films for q in film["study"].get("saved_queries", [])
                      if q.get("id") == saved_query_id), None)
        if saved is None:
            raise ValueError("所选项目没有这条保存的查询，请明确传入查询条件")
        query = {"id": saved_query_id, "name": saved.get("name", ""), "clauses": saved["clauses"]}
    if query is not None:
        clauses = validate_query_clauses(query.get("clauses") if isinstance(query, dict) else query)

        def needs_motion(rows):
            return any(c["field"] in ("stillness_min", "stillness_max")
                       or (c["field"] in ("all", "any", "not") and needs_motion(c.get("clauses", [])))
                       for c in rows)

        if needs_motion(clauses):
            raise ValueError("跨集项目没有保存运动数据；静止率条件只能用于当前项目")
    for film in films:
        shots = [s for s in film["shots"] if not s["excluded"]]
        film["eligible_count"] = len(shots)
        if query is not None:
            shots = query_matches({"shots": shots}, film["study"], film["metadata"], [], clauses)["shots"]
        film.update(shots=shots, durations=[s["duration_s"] for s in shots])
    rows = []
    for film in films:
        rows.append(dict(label=film["label"], path=film["path"], metadata=film["metadata"],
                         excluded_shots=film["excluded_shots"], complete=film["complete"],
                         **summary(film["durations"]),
                         **({"query_matched": len(film["shots"]),
                             "query_total": film["eligible_count"]} if query is not None else {})))
    pairs = []
    for a, b in combinations(range(len(films)), 2):
        first, second = (_inference_samples(films[i], inference_unit) for i in (a, b))
        enough = min(len(first), len(second)) >= 2
        test = mann_whitney(first, second) if enough else None
        dominance = cliffs_delta(first, second) if enough else None
        pairs.append(dict(first=films[a]["label"], second=films[b]["label"],
                          inference_unit=inference_unit, first_n=len(first), second_n=len(second),
                          mann_whitney=test, cliffs_delta=dominance,
                          note="" if enough else "每侧至少需要两个该单位的样本；本对只报告描述统计"))
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
            shared.append(dict(run, first_label=a["label"], second_label=b["label"],
                               classification="possible_material_reuse",
                               note="哈希共享提示可能的素材复用；不能作为构图相似的证据"))
    hierarchy = stratified_summary(films, inference_unit)
    aligned = manual_alignment(alignment or []) if alignment is not None else None
    query_summary = None
    if query is not None:
        query_summary = dict(name=query.get("name", query.get("id", "")) if isinstance(query, dict) else "",
                             clauses=clauses, films=[
                                 dict(label=film["label"], matched=len(film["shots"]),
                                      total=film["eligible_count"]) for film in films],
                             matched=sum(len(film["shots"]) for film in films),
                             total=sum(film["eligible_count"] for film in films),
                             strata=hierarchy)
    return dict(films=rows, pairs=pairs, groups=groups, group_by=group_by, episode_trend=trend,
                shared_sequences=shared, exclude_regions=exclude_regions,
                equal_weight=hierarchy, inference_unit=inference_unit, version_alignment=aligned,
                hashed_projects=len(hashed), query_summary=query_summary,
                notes=["时长单位秒；MSL 区间为次序统计量 95% 区间，剪辑率为每分钟镜头数（Poisson 95%）",
                       "Cliff's delta > 0 表示前者的镜头普遍更长",
                       "共享片段独立于查询范围，使用各项目保存的重复镜头哈希",
                       "每片等权重汇总与分层推断单位见 equal_weight；镜头数量不同不会自动增加影片权重",
                       "人工锚点把构图相似与素材复用分开；插入/删除镜头、裁剪与速度差异需人工注明",
                       "场景检验以每个已标注场景的平均镜头时长为样本；跨场景镜头归给覆盖最多帧的场景",
                       "影片或集为单位时，每个项目只有一个观测值；两片之间不计算显著性",
                       "跨集组合查询先排除区段，再对命中镜头汇总和检验；静止率只支持当前会话"])


def write_compare_csv(result, path):
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["label", "shot_count", "duration_s", "msl_s", "msl_ci95_s", "asl_s", "q1_s", "q3_s", "iqr_s",
              "cutting_rate_per_min", "cutting_rate_ci95_per_min", "excluded_shots", "query_matched",
              "query_total", "complete", "path"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in result["films"]:
            writer.writerow({k: row.get(k) for k in fields})
    return str(path.resolve())
