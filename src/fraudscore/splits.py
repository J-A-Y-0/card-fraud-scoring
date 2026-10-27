"""Ways of carving the development period into train and test.

The out-of-time window (Nov-Dec) is fixed and identical for every strategy, so the
strategies differ only in how honest their *test* number is.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from . import config


def periods(dates: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Boolean masks for (development, out-of-time) records; warm-up days are in neither."""
    dev = (dates >= config.MODEL_START) & (dates < config.OOT_START)
    oot = dates >= config.OOT_START
    return dev, oot


def random_split(dev: pd.DataFrame, seed: int):
    """The common default: shuffle records, stratified on the label."""
    return train_test_split(dev.index, test_size=0.3, stratify=dev["Fraud"], random_state=seed)


def card_split(dev: pd.DataFrame, seed: int):
    """Every card is entirely in train or entirely in test (about 30% of records in test)."""
    folds = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    fold_of = np.empty(len(dev), dtype=int)
    for k, (_, test_pos) in enumerate(folds.split(dev, dev["Fraud"], groups=dev["Cardnum"])):
        fold_of[test_pos] = k
    test = fold_of < 3
    return dev.index[~test], dev.index[test]


def temporal_split(dev: pd.DataFrame, seed: int = 0):
    """Train on Jan 15 - Aug 31, test on Sep - Oct. Deterministic; seed is ignored."""
    test = dev["Date"] >= config.TEMPORAL_TEST_START
    return dev.index[~test], dev.index[test]


STRATEGIES = {
    "random": random_split,
    "card_grouped": card_split,
    "temporal": temporal_split,
}
