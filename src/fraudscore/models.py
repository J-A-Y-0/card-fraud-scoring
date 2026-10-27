"""Feature filtering and model definitions."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .metrics import ks_statistic


def ks_filter(X: pd.DataFrame, y: pd.Series, keep: int = 40, max_corr: float = 0.95) -> pd.DataFrame:
    """Rank variables by univariate KS on the given (training) rows and drop near-duplicates.

    Returns a table of kept variables with their KS, in rank order.
    """
    ks = pd.Series({c: ks_statistic(X[c], y) for c in X.columns}).sort_values(ascending=False)
    ranked = X[ks.index].rank(pct=True)  # Spearman-style correlation, robust to heavy tails
    kept: list[str] = []
    for col in ks.index:
        if len(kept) == keep:
            break
        if kept and ranked[kept].corrwith(ranked[col]).abs().max() > max_corr:
            continue
        kept.append(col)
    return pd.DataFrame({"variable": kept, "ks": ks[kept].to_numpy()})


def _signed_log(X):
    return np.sign(X) * np.log1p(np.abs(X))


def make_model(name: str, seed: int = 0):
    if name == "boosted_trees":
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=100,
            l2_regularization=1.0, early_stopping=False, random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, min_samples_split=100, min_samples_leaf=20, max_features="sqrt",
            n_jobs=-1, random_state=seed,
        )
    if name == "logistic":
        return make_pipeline(
            FunctionTransformer(_signed_log), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)
        )
    raise ValueError(f"unknown model {name!r}")
