"""Point-in-time behavioral variables.

Every variable for a transaction uses only that transaction and the ones before it
(same day counts only if the record number is lower), so nothing leaks from the future.
Data is daily, so a "0-day" window means "earlier today, including this one".
"""

import numpy as np
import pandas as pd

ENTITIES = {
    "card": ["Cardnum"],
    "merch": ["Merchnum"],
    "card_merch": ["Cardnum", "Merchnum"],
    "card_zip": ["Cardnum", "Merch_zip"],
    "card_state": ["Cardnum", "Merch_state"],
    "merch_zip": ["Merchnum", "Merch_zip"],
}
WINDOWS = (0, 1, 3, 7, 14, 30)
VELOCITY_PAIRS = ((0, 7), (0, 14), (0, 30), (1, 7), (1, 14), (1, 30))


def _window_stats(day: np.ndarray, group: np.ndarray, amount: np.ndarray, window: int):
    """Count and sum over the trailing `window` days within each group.

    Arrays must be sorted by (group, day, record). For row i the window covers rows
    j <= i in the same group with day[j] >= day[i] - window.
    """
    key = group.astype(np.int64) * 100_000 + day
    start = np.searchsorted(key, key - window, side="left")
    idx = np.arange(len(key))
    csum = np.concatenate([[0.0], np.cumsum(amount)])
    count = idx - start + 1
    total = csum[idx + 1] - csum[start]
    return count, total


def _window_max(frame: pd.DataFrame, window: int) -> np.ndarray:
    rolled = frame.groupby("_g", sort=False).rolling(f"{window + 1}D", on="Date")["Amount"].max()
    return rolled.to_numpy()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row of variables per transaction, indexed like `df`, keyed by Recnum."""
    base = df[["Recnum", "Date", "Amount"]].copy()
    day = (base["Date"] - base["Date"].min()).dt.days.to_numpy()
    out = {"Recnum": df["Recnum"].to_numpy()}

    for name, cols in ENTITIES.items():
        g = df.groupby(cols, sort=False).ngroup().to_numpy()
        order = np.lexsort((df["Recnum"].to_numpy(), day, g))
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        gs, ds, amt = g[order], day[order], base["Amount"].to_numpy()[order]

        frame = pd.DataFrame({"_g": gs, "Date": base["Date"].to_numpy()[order], "Amount": amt})
        prev_same = np.r_[False, gs[1:] == gs[:-1]]
        gap = np.where(prev_same, np.r_[0, np.diff(ds)], 365).astype(float)
        out[f"{name}_days_since"] = gap[inv]

        counts = {}
        for w in WINDOWS:
            count, total = _window_stats(ds, gs, amt, w)
            counts[w] = count
            mean = total / count
            out[f"{name}_count_{w}"] = count[inv]
            out[f"{name}_total_{w}"] = total[inv]
            out[f"{name}_avg_{w}"] = mean[inv]
            out[f"{name}_max_{w}"] = _window_max(frame, w)[inv]
            out[f"{name}_actual_over_avg_{w}"] = (amt / mean)[inv]

        # Burst ratio: activity in the last 0-1 days vs the daily rate over a longer window.
        for short, long in VELOCITY_PAIRS:
            rate = counts[long] / (long + 1)
            out[f"{name}_velocity_{short}_by_{long}"] = (counts[short] / (short + 1) / rate)[inv]

    feats = pd.DataFrame(out, index=df.index)
    feats["amount"] = df["Amount"].to_numpy()
    feats["amount_cents_zero"] = (np.round(df["Amount"].to_numpy() * 100) % 100 == 0).astype(int)
    return feats
