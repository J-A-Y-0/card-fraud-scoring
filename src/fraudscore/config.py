"""Project-wide constants: paths, time boundaries, and evaluation settings."""

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("FRAUD_DATA_DIR", ROOT / "data" / "raw"))
ARTIFACTS_DIR = Path(os.environ.get("FRAUD_ARTIFACTS_DIR", ROOT / "artifacts"))

TRANSACTIONS_FILE = os.environ.get("FRAUD_TRANSACTIONS_FILE", "transactions.csv")
ZIP_FILE = os.environ.get("FRAUD_ZIP_FILE", "zip_code_database.csv")

# Variables look back up to 30 days, so the first two weeks are used to warm them up
# and are not scored or evaluated.
MODEL_START = pd.Timestamp("2010-01-15")
# Out-of-time holdout: the last two months, never touched by fitting, selection or thresholds.
OOT_START = pd.Timestamp("2010-11-01")
# Within the development period, the temporal split holds out Sep-Oct.
TEMPORAL_TEST_START = pd.Timestamp("2010-09-01")

ALERT_RATE = 0.03  # review budget: share of transactions an analyst team can look at
SEEDS = (0, 1, 2, 3, 4)
