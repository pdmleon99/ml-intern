import numpy as np


def bootstrap_metric_ci(y_true, y_pred, metric_fn, n_resamples: int = 1000, ci: float = 0.95, seed: int = 42) -> dict:
    """Bootstrap CI for a metric on held-out test-set predictions (resamples rows, not folds)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.RandomState(seed)
    raw_scores: list = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        try:
            raw_scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            continue
    if not raw_scores:
        return {"mean": None, "ci_low": None, "ci_high": None, "ci_level": ci}
    scores = np.array(raw_scores)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(scores, [alpha, 1 - alpha])
    return {
        "mean": float(scores.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_level": ci,
    }


def compare_to_baseline(model_fold_scores: list, baseline_fold_scores: list,
                         n_resamples: int = 2000, seed: int = 42) -> dict:
    """Paired bootstrap comparison of a model's CV fold scores against the naive baseline's —
    answers "how much better, and how confident are we?" instead of reporting a bare number.
    """
    m = np.asarray(model_fold_scores, dtype=float)
    b = np.asarray(baseline_fold_scores, dtype=float)
    n = min(len(m), len(b))
    if n == 0:
        return {
            "lift": None, "lift_pct": None, "ci_low": None, "ci_high": None,
            "probability_better_than_baseline": None, "confidence": "unknown",
        }
    m, b = m[:n], b[:n]
    diffs = m - b
    rng = np.random.RandomState(seed)
    boot_means = np.array([
        rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_resamples)
    ])
    lo, hi = np.quantile(boot_means, [0.025, 0.975])
    prob_better = float((boot_means > 0).mean())
    lift = float(m.mean() - b.mean())
    baseline_mean = float(b.mean())
    lift_pct = float(lift / abs(baseline_mean)) if baseline_mean != 0 else None

    if prob_better >= 0.975:
        confidence = "high"
    elif prob_better >= 0.85:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "lift": lift,
        "lift_pct": lift_pct,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "probability_better_than_baseline": prob_better,
        "confidence": confidence,
    }
