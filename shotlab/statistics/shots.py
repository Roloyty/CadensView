"""M1–M4: exact-time shot statistics, without decoding or Qt."""

from bisect import bisect_left
from dataclasses import asdict
from fractions import Fraction
from math import sqrt
from time import perf_counter

from .model import AnalysisScope, StatisticsParameters, build_context, contiguous_blocks
from .robust import summary as robust_summary


def grid(end, step):
    position = Fraction()
    while position < end:
        yield position
        position += step


def percentile(values, percent):
    position = Fraction(percent, 100) * (len(values) - 1)
    left = position.numerator // position.denominator
    fraction = position - left
    return values[left] + fraction * (values[min(left + 1, len(values) - 1)] - values[left])


def shot_statistics(context, parameters=StatisticsParameters()):
    parameters.validate()
    began = perf_counter()
    records = context.records
    durations = sorted(s.duration for s in records)
    count, total = len(records), sum(durations, Fraction())
    average = total / count if count else None
    deviation = sqrt(float(sum((d - average) ** 2 for d in durations) / count)) if count else None
    ranks = lambda reverse: [s.to_dict() for s in sorted(records, key=lambda s: (-s.duration if reverse else s.duration, s.start))[:parameters.rank_count]]
    longest, shortest = ranks(True), ranks(False)
    robust = robust_summary(durations) if count else {}
    overview = dict(shot_count=count, scope_duration_s=float(total), asl_s=float(average) if count else None,
                    msl_s=float(percentile(durations, 50)) if count else None, stddev_s=deviation,
                    cv=deviation / float(average) if average else None, longest_shots=longest, shortest_shots=shortest,
                    msl_ci95_s=robust.get("msl_ci95_s"), iqr_s=robust.get("iqr_s"), qcd=robust.get("qcd"),
                    cutting_rate_per_min=robust.get("cutting_rate_per_min"),
                    cutting_rate_ci95_per_min=robust.get("cutting_rate_ci95_per_min"))
    bin_width = Fraction(str(parameters.histogram_bin_s))
    histogram = []
    if durations:
        histogram = [dict(start_s=float(i * bin_width), end_s=float((i + 1) * bin_width), shot_ids=[],
                          count=0, duration_s=0) for i in range(int(durations[-1] // bin_width) + 1)]
        sums = [Fraction() for _ in histogram]
        for shot in records:
            i = int(shot.duration // bin_width)
            histogram[i]["shot_ids"].append(shot.shot_id)
            sums[i] += shot.duration
        for row, duration in zip(histogram, sums):
            n = len(row["shot_ids"])
            row.update(count=n, duration_s=float(duration), count_share=n / count, duration_share=float(duration / total))
    distribution = dict(duration_histogram=histogram, percentiles_s={f"p{p}": float(percentile(durations, p)) if count else None for p in (10, 25, 75, 90)})
    blocks = contiguous_blocks(records)
    cuts = [dict(boundary_frame=b.start_frame, left_frame=a.end_frame_exclusive - 1, right_frame=b.start_frame,
                 left_shot_id=a.shot_id, right_shot_id=b.shot_id, video_time_s=float(b.start))
            for block in blocks for a, b in zip(block, block[1:])]
    cut_times = [b.start for block in blocks for b in block[1:]]
    rolling = []
    for block in blocks:
        prefix = [Fraction()]
        for shot in block:
            prefix.append(prefix[-1] + shot.duration)
        for i, shot in enumerate(block):
            first, end = max(0, i - (parameters.rolling_shots - 1) // 2), min(len(block), i + parameters.rolling_shots // 2 + 1)
            rolling.append(dict(shot_id=shot.shot_id, video_time_s=float(shot.center), sample_count=end - first,
                                asl_s=float((prefix[end] - prefix[first]) / (end - first)),
                                first_shot_id=block[first].shot_id, last_shot_id=block[end - 1].shot_id))
    def window(first, end):
        first, end = max(Fraction(), first), min(context.duration, end)
        covered = sum((max(Fraction(), min(end, b[-1].end) - max(first, b[0].start)) for b in blocks), Fraction())
        left, right = bisect_left(cut_times, first), bisect_left(cut_times, end)
        return dict(start_time_s=float(first), end_time_s=float(end), effective_window_seconds=float(covered),
                    cut_count=right - left, cuts_per_minute=float((right - left) * 60 / covered) if covered else None), cuts[left:right]
    half = Fraction(str(parameters.density_window_s)) / 2
    density = [dict(window(t - half, t + half)[0], video_time_s=float(t)) for t in grid(context.duration, Fraction(str(parameters.time_step_s)))]
    segments = []
    for first in grid(context.duration, Fraction(str(parameters.segment_s))):
        end = min(context.duration, first + Fraction(str(parameters.segment_s)))
        data, _ = window(first, end)
        members = [s for s in records if first <= s.center < end]
        data.update(shot_count=len(members), shot_ids=[s.shot_id for s in members],
                    asl_s=float(sum((s.duration for s in members), Fraction()) / len(members)) if members else None)
        segments.append(data)
    width = min(context.duration, Fraction(str(parameters.dense_window_s)))
    last_start = context.duration - width
    # Starting at the first included cut cannot lose any cut from an earlier
    # window. Exact cut candidates prevent a sub-second burst falling between grid points.
    starts = sorted(set([*grid(last_start, Fraction(str(parameters.dense_step_s))), last_start,
                         *(t for t in cut_times if t <= last_start)]))
    candidates = []
    for first in starts:
        data, events = window(first, first + width)
        if events:
            candidates.append((first, dict(data, boundaries=events)))
    dense = []
    dense_starts = []
    for first, candidate in sorted(candidates, key=lambda x: (-x[1]["cut_count"], x[0])):
        if all(max(Fraction(), min(first + width, old + width) - max(first, old)) <
               Fraction(str(parameters.dense_max_overlap)) * width for old in dense_starts):
            dense.append(candidate); dense_starts.append(first)
            if len(dense) == parameters.dense_count:
                break
    runs = []
    threshold = Fraction(str(parameters.short_threshold_s))
    for block in blocks:
        group = []
        for shot in [*block, None]:
            if shot is not None and shot.duration < threshold:
                group.append(shot)
                continue
            if len(group) >= parameters.short_run_min:
                runs.append(dict(start_frame=group[0].start_frame, end_frame_exclusive=group[-1].end_frame_exclusive,
                                 start_time_s=float(group[0].start), end_time_s=float(group[-1].end),
                                 shot_count=len(group), duration_s=float(group[-1].end - group[0].start), shot_ids=[s.shot_id for s in group]))
            group = []
    methods = dict(asdict(parameters), percentiles="Hyndman-Fan type 7 (linear)", standard_deviation="population",
                   median_interval="order-statistic binomial 95%", cutting_rate="shots per minute, Byar Poisson 95%",
                   even_rolling_window="extra sample after center", density_denominator="covered time inside window",
                   segment_shot_assignment="full Shot by midpoint", dense_candidates="source-time grid, every exact cut, and final full window")
    return dict(metadata=context.metadata(methods), status="ok" if records else "empty", shots=[s.to_dict() for s in records],
                overview=overview, distribution=distribution,
                editing=dict(shot_duration_series=[s.to_dict() for s in records], rolling_asl_series=rolling,
                             boundaries=cuts, cut_density_series=density, time_segment_summary=segments),
                special=dict(longest_shots=longest, shortest_shots=shortest, densest_cut_windows=dense,
                             short_shot_runs=sorted(runs, key=lambda x: (-x["shot_count"], x["start_time_s"]))),
                elapsed_s=perf_counter() - began)


def compute_shot_statistics(project, index, scope=AnalysisScope(), parameters=StatisticsParameters()):
    return shot_statistics(build_context(project, index, scope), parameters)
