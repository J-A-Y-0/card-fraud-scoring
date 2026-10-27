"""Hyperparameter search with forward-chaining time folds.

Two folds, each training on everything before its test window:
  fold A: train Jan 15 - Jun 30, test Jul - Aug
  fold B: train Jan 15 - Aug 31, test Sep - Oct
The out-of-time window (Nov-Dec) is scored for the record but never used to choose.

Test windows hold only ~45-50 compromised cards each, so differences of a few points
are noise: in the first search 35 of 36 boosted-tree configs sat within one standard
error of the best, which makes the one-standard-error rule degenerate into "pick the
most constrained model". The pick is therefore the best mean test FDR across both
folds, and `within_one_se` reports how many configs are statistically indistinguishable.
"""

import itertools

import numpy as np
import pandas as pd

from . import config
from .metrics import fdr_at
from .models import ks_filter, make_model
from .splits import periods

FOLDS = {
    "A": (pd.Timestamp("2010-07-01"), pd.Timestamp("2010-09-01")),
    "B": (pd.Timestamp("2010-09-01"), config.OOT_START),
}

GRIDS = {
    "boosted_trees": {
        "learning_rate": [0.03, 0.1],
        "max_iter": [100, 300],
        "max_leaf_nodes": [7, 15, 31],
        "min_samples_leaf": [50, 200, 800],
    },
    "random_forest": {
        "max_depth": [8, None],
        "min_samples_leaf": [20, 100],
    },
    "logistic": {"C": [0.01, 0.1, 1.0]},
}


def fold_indices(df: pd.DataFrame):
    dev_mask, _ = periods(df["Date"])
    for name, (start, end) in FOLDS.items():
        train = df.index[dev_mask & (df["Date"] < start)]
        test = df.index[(df["Date"] >= start) & (df["Date"] < end)]
        yield name, train, test


def fdr_card_se(y, score, cards, rate=config.ALERT_RATE, n=1000, seed=0) -> float:
    """Bootstrap standard error of FDR, resampling compromised cards with scores held fixed."""
    y = np.asarray(y)
    k = int(round(len(y) * rate))
    flagged = np.zeros(len(y), bool)
    flagged[np.argsort(-np.asarray(score), kind="stable")[:k]] = True
    per_card = pd.DataFrame({"card": np.asarray(cards)[y == 1], "hit": flagged[y == 1]})
    agg = per_card.groupby("card")["hit"].agg(["sum", "size"]).to_numpy()
    pick = np.random.default_rng(seed).integers(0, len(agg), (n, len(agg)))
    return float((agg[pick, 0].sum(1) / agg[pick, 1].sum(1)).std())


def _set_params(model, params):
    # Logistic is a pipeline; its C lives on the last step.
    if hasattr(model, "steps"):
        model.steps[-1][1].set_params(**params)
    else:
        model.set_params(**params)
    return model


def evaluate(df, feats, model_name, params, n_vars, rankings, seed=0) -> dict:
    """Mean train/test FDR@3% over the time folds, plus OOT for the record."""
    X, y = feats.drop(columns="Recnum"), df["Fraud"]
    _, oot_mask = periods(df["Date"])
    oot_idx = df.index[oot_mask]
    row = {"model": model_name, **params, "n_vars": n_vars}
    tests, ses = [], []
    for name, train, test in fold_indices(df):
        cols = rankings[name][:n_vars]
        model = _set_params(make_model(model_name, seed), params).fit(X.loc[train, cols], y.loc[train])
        s_test = model.predict_proba(X.loc[test, cols])[:, 1]
        row[f"train_{name}"] = fdr_at(y.loc[train], model.predict_proba(X.loc[train, cols])[:, 1])
        row[f"test_{name}"] = fdr_at(y.loc[test], s_test)
        row[f"oot_{name}"] = fdr_at(y.loc[oot_idx], model.predict_proba(X.loc[oot_idx, cols])[:, 1])
        tests.append(row[f"test_{name}"])
        ses.append(fdr_card_se(y.loc[test], s_test, df.loc[test, "Cardnum"]))
    row["train"] = np.mean([row["train_A"], row["train_B"]])
    row["test"] = np.mean(tests)
    row["test_se"] = float(np.sqrt(np.sum(np.square(ses))) / len(ses))
    row["oot"] = row["oot_B"]  # fold B's model is the one trained closest to the OOT window
    row["train_minus_test"] = row["train"] - row["test"]
    return row


def variable_rankings(df, feats, keep=40) -> dict[str, list[str]]:
    """KS-filter ranking computed separately on each fold's training rows."""
    X, y = feats.drop(columns="Recnum"), df["Fraud"]
    return {name: ks_filter(X.loc[train], y.loc[train], keep=keep)["variable"].tolist()
            for name, train, _ in fold_indices(df)}


QUICK_GRIDS = {
    "boosted_trees": {"learning_rate": [0.1], "max_iter": [100], "max_leaf_nodes": [7, 31], "min_samples_leaf": [50]},
    "random_forest": {"max_depth": [None], "min_samples_leaf": [20]},
    "logistic": {"C": [0.1]},
}


def search(df, feats, var_counts=(10, 20, 30, 40, 50, 60), quick: bool = False) -> tuple[pd.DataFrame, dict]:
    if quick:
        var_counts = (20, 40)
    grids = QUICK_GRIDS if quick else GRIDS
    rankings = variable_rankings(df, feats, keep=max(var_counts))
    rows = []
    default = {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15, "min_samples_leaf": 100}
    for n in var_counts:  # stage 1: how many variables
        rows.append({"stage": "n_vars", **evaluate(df, feats, "boosted_trees", default, n, rankings)})
        print(f"n_vars={n} test={rows[-1]['test']:.3f}", flush=True)
    stage1 = pd.DataFrame(rows)
    n_best = int(stage1.loc[stage1["test"].idxmax(), "n_vars"])
    for model_name, grid in grids.items():  # stage 2: hyperparameters at that variable count
        for values in itertools.product(*grid.values()):
            params = dict(zip(grid.keys(), values))
            rows.append({"stage": "grid", **evaluate(df, feats, model_name, params, n_best, rankings)})
            print(model_name, params, f"train={rows[-1]['train']:.3f} test={rows[-1]['test']:.3f}", flush=True)
    return pd.DataFrame(rows), rankings


def within_one_se(results: pd.DataFrame) -> pd.Series:
    """Rows whose mean test FDR is within one SE of the best (statistically tied with it)."""
    best = results.loc[results["test"].idxmax()]
    return results["test"] >= best["test"] - best["test_se"]


def complexity_curve(df, feats, rankings, n_vars, base: dict, param: str, values) -> pd.DataFrame:
    """Train vs test FDR as one boosted-tree parameter grows (for the overfitting plot)."""
    rows = []
    for v in values:
        r = evaluate(df, feats, "boosted_trees", {**base, param: v}, n_vars, rankings)
        rows.append({"param": param, "value": v, "train": r["train"], "test": r["test"], "test_se": r["test_se"]})
    return pd.DataFrame(rows)
