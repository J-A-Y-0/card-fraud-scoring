"""Synthetic card transactions with the same schema and quirks as the real file.

Lets anyone run the full pipeline without the (non-redistributable) dataset, and gives CI
an end-to-end smoke test. The structure is what matters for the method, not realism:
cards with habits and favorite merchants, compromised cards whose fraud arrives in bursts
at unfamiliar merchants, and the data-quality problems the cleaning step exists to fix.
"""

from pathlib import Path

import numpy as np
import pandas as pd

STATES = ["CA", "TX", "NY", "VA", "TN", "IL", "MD", "GA", "PA", "NJ", "NC", "WA", "FL", "OH", "CO"]


def zip_table(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for i, st in enumerate(STATES):
        for j in range(20):
            rows.append({"zip": 10000 + i * 1000 + j * 7, "state": st, "irs_estimated_population": int(rng.integers(100, 60000))})
    return pd.DataFrame(rows)


def generate(n_cards: int = 500, n_merchants: int = 1500, n_compromised: int = 70, seed: int = 0):
    """Return (transactions, zip_codes) DataFrames in the raw file layout."""
    rng = np.random.default_rng(seed)
    zips = zip_table(rng)
    merch = pd.DataFrame({
        "Merchnum": [str(9000000 + i) for i in range(n_merchants)],
        "desc": [f"MERCHANT {i:04d}" for i in range(n_merchants)],
    })
    loc = zips.sample(n_merchants, replace=True, random_state=seed).reset_index(drop=True)
    merch["state"], merch["zip"] = loc["state"], loc["zip"].astype(float)
    popularity = rng.pareto(1.2, n_merchants) + 1
    popularity /= popularity.sum()

    days = pd.date_range("2010-01-01", "2010-12-31")
    rows = []
    for c in range(n_cards):
        card = str(5142000000 + c)
        rate = rng.gamma(2.0, 0.12)                       # purchases per day
        favorites = rng.choice(n_merchants, 12, p=popularity)
        mean_amt = rng.lognormal(4.5, 0.8)
        for d in np.flatnonzero(rng.random(len(days)) < 1 - np.exp(-rate)):
            for _ in range(1 + rng.poisson(rate / 2)):
                m = favorites[rng.integers(len(favorites))] if rng.random() < 0.8 else rng.choice(n_merchants, p=popularity)
                rows.append((card, days[d], m, round(rng.lognormal(np.log(mean_amt), 0.9), 2), 0))

    # Compromised cards: a run of fraud at unfamiliar merchants, most of it in the first days.
    for c in rng.choice(n_cards, n_compromised, replace=False):
        card = str(5142000000 + c)
        start = rng.integers(20, len(days) - 3)
        n_fraud = 1 if rng.random() < 0.3 else int(rng.integers(2, 40))
        offsets = np.sort(np.minimum(rng.exponential(4.0, n_fraud).astype(int), 30))
        for off in offsets:
            if start + off < len(days):
                m = rng.choice(n_merchants)
                rows.append((card, days[start + off], m, round(rng.lognormal(6.0, 1.0), 2), 1))

    t = pd.DataFrame(rows, columns=["Cardnum", "Date", "m", "Amount", "Fraud"]).sort_values(["Date"], kind="stable")
    t = t.reset_index(drop=True)
    t["Merchnum"] = merch.loc[t["m"], "Merchnum"].to_numpy()
    t["Merch description"] = merch.loc[t["m"], "desc"].to_numpy()
    t["Merch state"] = merch.loc[t["m"], "state"].to_numpy()
    t["Merch zip"] = merch.loc[t["m"], "zip"].to_numpy()
    t["Transtype"] = "P"

    # The data-quality problems the cleaning step handles.
    n = len(t)
    pick = lambda share: rng.random(n) < share
    t.loc[pick(0.03), "Merchnum"] = np.nan
    t.loc[pick(0.003), "Merchnum"] = "0"
    t.loc[pick(0.012), "Merch state"] = np.nan
    t.loc[pick(0.002), "Merch state"] = "297"
    t.loc[pick(0.045), "Merch zip"] = np.nan
    t.loc[pick(0.004) & (t["Fraud"] == 0), "Transtype"] = "A"
    t.loc[t.index[rng.integers(n)], ["Amount", "Fraud"]] = [3_000_000.0, 0]

    t.insert(0, "Recnum", np.arange(1, n + 1))
    t["Date"] = t["Date"].dt.strftime("%-m/%-d/%y")
    cols = ["Recnum", "Cardnum", "Date", "Merchnum", "Merch description", "Merch state", "Merch zip", "Transtype", "Amount", "Fraud"]
    return t[cols], zips


def write(out_dir: Path, **kwargs) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    t, zips = generate(**kwargs)
    t.to_csv(out_dir / "transactions.csv", index=False)
    zips.to_csv(out_dir / "zip_code_database.csv", index=False)
    return out_dir
