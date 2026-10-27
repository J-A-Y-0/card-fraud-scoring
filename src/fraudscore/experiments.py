"""Validation audit: how much does the choice of test split flatter the model?

For each split strategy the same model is trained on the strategy's train rows and
scored on its test rows and on the shared out-of-time window. A trustworthy test
split is one whose number is close to the out-of-time number.
"""

import pandas as pd

from . import config
from .metrics import fdr_at
from .models import ks_filter, make_model
from .splits import STRATEGIES, periods


def leakage_share(df: pd.DataFrame, train_idx, test_idx) -> float:
    """Share of test frauds on a card that also has a fraud in train."""
    fraud_cards_train = set(df.loc[train_idx].query("Fraud == 1")["Cardnum"])
    test_frauds = df.loc[test_idx].query("Fraud == 1")
    return float(test_frauds["Cardnum"].isin(fraud_cards_train).mean())


def fit_and_score(X, y, train_idx, test_idx, oot_idx, model_name, seed, select=None):
    cols = list(X.columns)
    if select:
        cols = ks_filter(X.loc[train_idx], y.loc[train_idx], keep=select)["variable"].tolist()
    model = make_model(model_name, seed).fit(X.loc[train_idx, cols], y.loc[train_idx])
    out = {}
    for part, idx in (("train", train_idx), ("test", test_idx), ("oot", oot_idx)):
        out[part] = fdr_at(y.loc[idx], model.predict_proba(X.loc[idx, cols])[:, 1], config.ALERT_RATE)
    return out, model, cols


def run_split_gap(df: pd.DataFrame, feats: pd.DataFrame, seeds=config.SEEDS) -> pd.DataFrame:
    dev_mask, oot_mask = periods(df["Date"])
    dev, oot_idx = df[dev_mask], df.index[oot_mask]
    y = df["Fraud"]
    X = feats.drop(columns="Recnum")
    rows = []
    for model_name in ("boosted_trees", "random_forest"):  # same 40 KS-filtered variables for both
        for strategy, split in STRATEGIES.items():
            for seed in seeds:
                train_idx, test_idx = split(dev, seed)
                scores, _, _ = fit_and_score(X, y, train_idx, test_idx, oot_idx, model_name, seed, 40)
                rows.append({
                    "model": model_name, "split": strategy, "seed": seed,
                    **scores, "test_frauds_on_train_fraud_cards": leakage_share(df, train_idx, test_idx),
                })
                print(f"{model_name:14s} {strategy:13s} seed={seed} "
                      f"train={scores['train']:.3f} test={scores['test']:.3f} oot={scores['oot']:.3f}", flush=True)
    return pd.DataFrame(rows)


def summarize_split_gap(results: pd.DataFrame) -> pd.DataFrame:
    g = results.groupby(["model", "split"], sort=False)
    summary = g[["train", "test", "oot", "test_frauds_on_train_fraud_cards"]].mean()
    summary[["test_sd", "oot_sd"]] = g[["test", "oot"]].std().to_numpy()
    summary["test_minus_oot"] = summary["test"] - summary["oot"]
    return summary.round(3).reset_index()
