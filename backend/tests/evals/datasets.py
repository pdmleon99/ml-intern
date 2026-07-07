"""Synthetic 'golden' datasets with deliberately planted issues — used to check whether the
agents actually catch/handle the thing they're supposed to, instead of eyeballing real data."""

import numpy as np
import pandas as pd


def leakage_dataset(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """A column that's essentially a copy of the target — Planner should drop it."""
    rng = np.random.RandomState(seed)
    f1 = rng.randn(n)
    f2 = rng.randn(n)
    target = (f1 + rng.randn(n) * 0.5 > 0).astype(int)
    return pd.DataFrame({
        "f1": f1, "f2": f2,
        "leak_col": target + rng.randn(n) * 0.01,
        "target": target,
    })


def no_signal_dataset(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Features are pure noise, unrelated to the target — no model should beat baseline."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "f1": rng.randn(n), "f2": rng.randn(n), "f3": rng.randn(n),
        "target": rng.randint(0, 2, n),
    })


def severe_imbalance_dataset(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """99/1 class split — a real signal exists, but accuracy alone would be misleading."""
    rng = np.random.RandomState(seed)
    f1 = rng.randn(n)
    y = np.concatenate([np.zeros(int(n * 0.99)), np.ones(n - int(n * 0.99))]).astype(int)
    y = rng.permutation(y)
    f1 = f1 + y * 1.5  # weak real signal for the minority class
    return pd.DataFrame({"f1": f1, "f2": rng.randn(n), "target": y})


def id_column_dataset(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """An obvious sequential ID column plus real continuous features."""
    rng = np.random.RandomState(seed)
    f1 = rng.uniform(18, 80, n)  # e.g. "age" — continuous, near-unique, but NOT an ID
    target = (f1 > 45).astype(int)
    return pd.DataFrame({
        "row_id": range(1, n + 1),
        "age": f1,
        "target": target,
    })
