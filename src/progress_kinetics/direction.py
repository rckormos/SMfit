"""One robust sign vote per informative trace, after preprocessing."""

import numpy as np
from scipy.stats import theilslopes


def detect_direction(data, indices):
    c = data.config
    votes = []
    details = []
    for index in indices:
        mask = data.used[index]
        t, y = data.time[index, mask], data.signal[index, mask]
        order = np.argsort(t, kind="stable")
        t, y = t[order], y[order]
        # Cap pairwise-slope cost deterministically for long acquisitions.
        pick = np.unique(np.linspace(0, len(t) - 1, min(80, len(t))).astype(int))
        slope, intercept, _, _ = theilslopes(y[pick], t[pick])
        detrended = y - slope * t - intercept
        # Difference MAD estimates point noise without interpreting smooth curvature as noise.
        diff = np.diff(detrended)
        noise = 1.4826 * np.median(np.abs(diff - np.median(diff))) / np.sqrt(2)
        window = max(1, len(y) // 5)
        change = float(np.median(y[-window:]) - np.median(y[:window]))
        threshold = max(c.direction_min_change_AU, c.direction_noise_multiplier * noise)
        informative = abs(change) >= threshold and slope * change > 0
        vote = int(np.sign(change)) if informative else 0
        if vote:
            votes.append(vote)
        details.append(
            {
                "plate": data.traces[index]["plate"],
                "well": data.traces[index]["well"],
                "slope_AU_per_s": float(slope),
                "window_change_AU": change,
                "noise_AU": float(noise),
                "threshold_AU": float(threshold),
                "vote": vote,
            }
        )
    positives, negatives = votes.count(1), votes.count(-1)
    agreement = max(positives, negatives) / len(votes) if votes else 0
    report = {
        "informative": len(votes),
        "flat_or_noisy": len(indices) - len(votes),
        "increasing_votes": positives,
        "decreasing_votes": negatives,
        "conflicting_votes": min(positives, negatives),
        "agreement": agreement,
        "minimum_traces": c.direction_min_traces,
        "required_agreement": c.direction_agreement,
        "trace_votes": details,
        "source": "auto",
    }
    if c.signal_direction != "auto":
        direction = 1 if c.signal_direction == "increasing" else -1
        report["source"] = "override"
    elif len(votes) >= c.direction_min_traces and agreement >= c.direction_agreement:
        direction = 1 if positives > negatives else -1
    else:
        direction = None
    report["direction"] = (
        "ambiguous" if direction is None else "increasing" if direction == 1 else "decreasing"
    )
    return direction, report
