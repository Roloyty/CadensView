"""Distribution-free summaries for skewed shot-length data (Redfern 2012, 2014, 2022).

The median and quartiles describe a right-skewed shot-length distribution
better than the mean; cutting rate is reported as a rate with its own interval
instead of being read off the ASL. Nothing here needs SciPy.
"""

from math import ceil, erf, floor, sqrt

Z95 = 1.959963984540054


def quantile(sorted_values, q):
    """Hyndman-Fan type 7, matching shots.percentile for floats."""
    if not sorted_values:
        return None
    position = q * (len(sorted_values) - 1)
    left = floor(position)
    right = min(left + 1, len(sorted_values) - 1)
    return float(sorted_values[left] + (position - left) * (sorted_values[right] - sorted_values[left]))


def median_ci(sorted_values, z=Z95):
    """Order-statistic (binomial) 95% interval for the median; no distribution assumed."""
    n = len(sorted_values)
    if n < 6:
        return None
    half = z * sqrt(n) / 2
    lower = max(1, floor(n / 2 - half))
    upper = min(n, ceil(n / 2 + half + 1))
    return [float(sorted_values[lower - 1]), float(sorted_values[upper - 1])]


def poisson_ci(count, z=Z95):
    """Byar's approximation to the exact Poisson interval for an event count."""
    if count < 0:
        raise ValueError("count must be non-negative")
    lower = 0.0 if count == 0 else count * (1 - 1 / (9 * count) - z / (3 * sqrt(count))) ** 3
    upper = (count + 1) * (1 - 1 / (9 * (count + 1)) + z / (3 * sqrt(count + 1))) ** 3
    return max(0.0, lower), upper


def cutting_rate(shot_count, seconds):
    """Shots per minute (Redfern's cutting rate) with a Poisson 95% interval."""
    if not shot_count or not seconds or seconds <= 0:
        return None, None
    minutes = seconds / 60
    lower, upper = poisson_ci(shot_count)
    return shot_count / minutes, [lower / minutes, upper / minutes]


def summary(durations):
    """Robust summary of a list of shot durations in seconds."""
    values = sorted(float(d) for d in durations)
    n = len(values)
    if not n:
        return dict(shot_count=0)
    total = sum(values)
    q1, q2, q3 = (quantile(values, q) for q in (.25, .5, .75))
    rate, rate_ci = cutting_rate(n, total)
    return dict(
        shot_count=n, duration_s=total, asl_s=total / n, msl_s=q2, msl_ci95_s=median_ci(values),
        q1_s=q1, q3_s=q3, iqr_s=q3 - q1, qcd=(q3 - q1) / (q3 + q1) if q3 + q1 else None,
        cutting_rate_per_min=rate, cutting_rate_ci95_per_min=rate_ci,
        deciles_s=[quantile(values, i / 10) for i in range(1, 10)])


def _normal_sf(x):
    return 0.5 * (1 - erf(x / sqrt(2)))


def mann_whitney(first, second):
    """Two-sided Mann-Whitney U with tie correction (normal approximation)."""
    n1, n2 = len(first), len(second)
    if not n1 or not n2:
        return None
    pooled = sorted([(float(v), 0) for v in first] + [(float(v), 1) for v in second])
    ranks = [0.0] * len(pooled)
    ties = 0.0
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = rank
        size = j - i + 1
        ties += size ** 3 - size
        i = j + 1
    rank_sum = sum(r for r, (_, group) in zip(ranks, pooled) if group == 0)
    u1 = rank_sum - n1 * (n1 + 1) / 2
    n = n1 + n2
    variance = n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1))) if n > 1 else 0
    if variance <= 0:
        return dict(u=u1, z=0.0, p_two_sided=1.0)
    z = (u1 - n1 * n2 / 2) / sqrt(variance)
    return dict(u=u1, z=z, p_two_sided=min(1.0, 2 * _normal_sf(abs(z))))


def cliffs_delta(first, second):
    """Dominance statistic P(X>Y) - P(X<Y), computed by merging sorted lists."""
    a, b = sorted(float(v) for v in first), sorted(float(v) for v in second)
    if not a or not b:
        return None
    greater = less = 0
    lo = hi = 0
    for x in a:
        while lo < len(b) and b[lo] < x:
            lo += 1
        while hi < len(b) and b[hi] <= x:
            hi += 1
        greater += lo
        less += len(b) - hi
    delta = (greater - less) / (len(a) * len(b))
    size = abs(delta)
    magnitude = "negligible" if size < .147 else "small" if size < .33 else "medium" if size < .474 else "large"
    return dict(delta=delta, magnitude=magnitude)


def cohen_kappa(pairs):
    """Cohen's kappa for (label_a, label_b) pairs plus the confusion matrix."""
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    n = len(pairs)
    if not n:
        return dict(n=0, kappa=None, agreement=None, confusion={})
    labels = sorted({x for pair in pairs for x in pair})
    observed = sum(a == b for a, b in pairs) / n
    expected = sum((sum(a == label for a, _ in pairs) / n) * (sum(b == label for _, b in pairs) / n)
                   for label in labels)
    kappa = None if expected == 1 else (observed - expected) / (1 - expected)
    confusion = {a: {b: sum(1 for x, y in pairs if x == a and y == b) for b in labels} for a in labels}
    return dict(n=n, kappa=kappa, agreement=observed, confusion=confusion)
