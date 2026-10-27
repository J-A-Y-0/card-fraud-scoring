"""Loading and cleaning of the raw card transaction file.

Cleaning never looks at the Fraud label, so it can safely use the whole file to
resolve merchant identities and locations.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN",
    "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
    "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT",
    "VT", "VA", "WA", "WV", "WI", "WY", "PR", "VI", "GU", "AS", "MP",
}
MAX_AMOUNT = 1_000_000  # one $3.1M record is a data error, not a purchase
UNKNOWN = "UNK"
FOREIGN = "FOREIGN"


def load_raw(data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(
        data_dir / config.TRANSACTIONS_FILE,
        dtype={"Cardnum": str, "Merchnum": str, "Merch state": str},
    )
    df = df.rename(
        columns={
            "Merch description": "Merch_desc",
            "Merch state": "Merch_state",
            "Merch zip": "Merch_zip",
        }
    )
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
    return df


def load_zip_reference(data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    z = pd.read_csv(data_dir / config.ZIP_FILE, usecols=["zip", "state", "irs_estimated_population"])
    z["zip"] = z["zip"].astype(int).map("{:05d}".format)
    return z


def apply_exclusions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep purchases only and drop amount errors. Returns (kept, exclusion log)."""
    rules = {
        "Non-purchase transaction type (A, D, Y)": df["Transtype"] != "P",
        f"Amount above ${MAX_AMOUNT:,.0f}": df["Amount"] > MAX_AMOUNT,
    }
    log, dropped = [], pd.Series(False, index=df.index)
    for name, mask in rules.items():
        new = mask & ~dropped
        log.append({"rule": name, "records": int(new.sum()), "fraud_records": int(df.loc[new, "Fraud"].sum())})
        dropped |= mask
    return df.loc[~dropped].reset_index(drop=True), pd.DataFrame(log)


def _mode_map(df: pd.DataFrame, key: str, value: str) -> pd.Series:
    """Most frequent non-null `value` for each `key` (ties broken alphabetically)."""
    known = df.dropna(subset=[key, value])
    counts = known.groupby([key, value]).size().reset_index(name="n")
    counts = counts.sort_values(["n", value], ascending=[False, True])
    return counts.drop_duplicates(key).set_index(key)[value]


def _fill_from(df: pd.DataFrame, target: str, key: str, log: list, step: str) -> None:
    missing = df[target].isna()
    filled = df.loc[missing, key].map(_mode_map(df, key, target))
    df.loc[missing, target] = filled
    log.append({"field": target, "step": step, "records": int(filled.notna().sum())})


def clean(df: pd.DataFrame, zips: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute merchant number, state and ZIP. Returns (clean data, imputation log)."""
    df = df.copy()
    log: list[dict] = []

    # Merchant number: '0' is a placeholder, not a merchant.
    df.loc[df["Merchnum"] == "0", "Merchnum"] = np.nan
    log.append({"field": "Merchnum", "step": "missing or '0' before imputation", "records": int(df["Merchnum"].isna().sum())})
    _fill_from(df, "Merchnum", "Merch_desc", log, "most common Merchnum for the same description")
    still = df["Merchnum"].isna()
    df.loc[still, "Merchnum"] = "D:" + df.loc[still, "Merch_desc"].str.strip()
    log.append({"field": "Merchnum", "step": "new id created from description", "records": int(still.sum())})

    # ZIP: normalize to 5-digit strings.
    df["Merch_zip"] = df["Merch_zip"].map(lambda z: f"{int(z):05d}" if pd.notna(z) else np.nan).astype(object)
    zip_state = zips.set_index("zip")["state"]

    # State: anything outside US states/territories is a foreign or garbled code.
    raw_state = df["Merch_state"].str.strip().str.upper()
    foreign = raw_state.notna() & ~raw_state.isin(US_STATES)
    df["Merch_state"] = raw_state.where(~foreign, FOREIGN)
    log.append({"field": "Merch_state", "step": "non-US code mapped to FOREIGN", "records": int(foreign.sum())})
    missing = df["Merch_state"].isna()
    from_zip = df.loc[missing, "Merch_zip"].map(zip_state)
    df.loc[missing, "Merch_state"] = from_zip
    log.append({"field": "Merch_state", "step": "looked up from ZIP", "records": int(from_zip.notna().sum())})
    _fill_from(df, "Merch_state", "Merchnum", log, "most common state for the same merchant")
    _fill_from(df, "Merch_state", "Merch_desc", log, "most common state for the same description")

    # ZIP imputation, most specific source first.
    _fill_from(df, "Merch_zip", "Merchnum", log, "most common ZIP for the same merchant")
    _fill_from(df, "Merch_zip", "Merch_desc", log, "most common ZIP for the same description")
    biggest_zip = (
        zips.sort_values("irs_estimated_population", ascending=False).drop_duplicates("state").set_index("state")["zip"]
    )
    missing = df["Merch_zip"].isna()
    by_state = df.loc[missing, "Merch_state"].map(biggest_zip)
    df.loc[missing, "Merch_zip"] = by_state
    log.append({"field": "Merch_zip", "step": "most populated ZIP in the state", "records": int(by_state.notna().sum())})

    for col in ("Merch_state", "Merch_zip"):
        rest = df[col].isna()
        df.loc[rest, col] = UNKNOWN
        log.append({"field": col, "step": f"left as {UNKNOWN}", "records": int(rest.sum())})

    return df, pd.DataFrame(log)


def load_clean(data_dir: Path = config.DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Raw file -> (clean transactions, exclusion log, imputation log)."""
    kept, excl = apply_exclusions(load_raw(data_dir))
    cleaned, imp = clean(kept, load_zip_reference(data_dir))
    return cleaned, excl, imp
