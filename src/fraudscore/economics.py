"""Turning replay outcomes into money, and money into a cutoff and a staffing level.

All cost inputs are explicit assumptions with ranges. They are placeholders until the
program owner confirms them; nothing here is a sourced benchmark.
"""

from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd

from . import value

PRODUCTIVE_HOURS_PER_ANALYST_DAY = 6.0


@dataclass(frozen=True)
class Assumptions:
    minutes_per_review: float = 10.0     # analyst time to work one alert
    analyst_cost_per_hour: float = 50.0  # fully loaded
    loss_share: float = 0.8              # share of unstopped fraud $ the program actually loses
    reissue_cost: float = 15.0           # per blocked card: new plastic, shipping, admin
    false_alert_cost: float = 5.0        # cardholder contact to clear a false alert
    declined_purchase_cost: float = 10.0  # each legitimate purchase declined on a blocked card
    reissue_lag_days: int = 5            # days until the replacement card arrives

    @property
    def review_cost(self) -> float:
        return self.minutes_per_review / 60 * self.analyst_cost_per_hour


RANGES = {
    "minutes_per_review": (5.0, 20.0),
    "analyst_cost_per_hour": (35.0, 75.0),
    "loss_share": (0.5, 1.0),
    "reissue_cost": (10.0, 30.0),
    "false_alert_cost": (0.0, 15.0),
    "declined_purchase_cost": (2.0, 25.0),
    "reissue_lag_days": (3, 7),
}


def net_value(summary: dict, a: Assumptions) -> dict:
    benefit = a.loss_share * summary["fraud_dollars_prevented"]
    costs = {
        "review": a.review_cost * summary["alerts"],
        "reissue": a.reissue_cost * summary["fraud_cards_detected"],
        "false_alert": a.false_alert_cost * summary["false_alerts"],
        "declined_purchases": a.declined_purchase_cost * summary["legit_txns_declined"],
    }
    total = sum(costs.values())
    return {"benefit": benefit, **{f"cost_{k}": v for k, v in costs.items()}, "cost_total": total,
            "net_value": benefit - total}


def sweep(txns, score, ref_score, alert_rates, a: Assumptions, policy="next_day_block", capacity=None):
    """Net value for each target alert rate. Cutoffs come from `ref_score` quantiles."""
    rows = []
    for r in alert_rates:
        cutoff = float(np.quantile(ref_score, 1 - r))
        out = value.replay(txns, score, cutoff, policy, capacity, a.reissue_lag_days)
        s = value.summarize(out)
        rows.append({"target_alert_rate": r, "cutoff": cutoff, **s, **net_value(s, a)})
    return pd.DataFrame(rows)


def staffing(alerts: int, days: int, a: Assumptions) -> dict:
    hours = alerts * a.minutes_per_review / 60 / days
    return {"alerts_per_day": alerts / days, "analyst_hours_per_day": hours,
            "analyst_fte": hours / PRODUCTIVE_HOURS_PER_ANALYST_DAY}


def tornado(txns, score, cutoff, a: Assumptions, policy="next_day_block") -> pd.DataFrame:
    """Net value when one assumption at a time moves to the low or high end of its range."""
    cache = {}

    def nv(assume):
        lag = assume.reissue_lag_days
        if lag not in cache:
            cache[lag] = value.summarize(value.replay(txns, score, cutoff, policy, None, lag))
        return net_value(cache[lag], assume)["net_value"]

    base = nv(a)
    rows = []
    for name, (lo, hi) in RANGES.items():
        v_lo, v_hi = nv(replace(a, **{name: lo})), nv(replace(a, **{name: hi}))
        rows.append({"assumption": name, "low": lo, "high": hi, "default": getattr(a, name),
                     "net_at_low": v_lo, "net_at_high": v_hi, "swing": abs(v_hi - v_lo)})
    return pd.DataFrame(rows).sort_values("swing", ascending=False).assign(base=base)


def monte_carlo(txns, score, cutoff, a: Assumptions, n=5000, seed=0, policy="next_day_block",
                vary_assumptions: bool = True) -> pd.DataFrame:
    """Joint uncertainty: assumptions drawn from triangular(low, default, high) and
    compromised cards resampled with replacement (sampling noise in the fraud side).
    With vary_assumptions=False only the cards are resampled."""
    rng = np.random.default_rng(seed)
    lo_lag, hi_lag = RANGES["reissue_lag_days"]
    outcomes = {lag: value.replay(txns, score, cutoff, policy, None, lag) for lag in range(lo_lag, hi_lag + 1)}
    per_lag = {}
    for lag, out in outcomes.items():
        s = value.summarize(out)
        f = out[out["Fraud"] == 1]
        stopped = f["status"].isin(value.STOPPED)
        cards = pd.DataFrame({
            "prevented": f[stopped].groupby("Cardnum")["Amount"].sum(),
            "detected": f["status"].isin(["alert_fraud", "declined_at_alert"]).groupby(f["Cardnum"]).any(),
        }).fillna(0.0)
        per_lag[lag] = (s, cards["prevented"].to_numpy(), cards["detected"].to_numpy().astype(float))

    rows = []
    for _ in range(n):
        draw = {}
        for f_ in fields(Assumptions) if vary_assumptions else ():
            lo, hi = RANGES[f_.name]
            mode = getattr(a, f_.name)
            draw[f_.name] = int(rng.integers(lo, hi + 1)) if f_.name == "reissue_lag_days" else rng.triangular(lo, mode, hi)
        assume = replace(a, **draw)
        s, prevented, detected = per_lag[assume.reissue_lag_days]
        pick = rng.integers(0, len(prevented), len(prevented))
        s = {**s, "fraud_dollars_prevented": prevented[pick].sum(), "fraud_cards_detected": detected[pick].sum()}
        rows.append({**draw, **net_value(s, assume)})
    return pd.DataFrame(rows)
