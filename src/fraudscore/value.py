"""Card-stop value: what an alert is worth when fraud comes in bursts.

Fraud in this data arrives as runs of transactions on a compromised card. A confirmed
alert lets the issuer block the card, which prevents the rest of the run. Scoring a
model by the dollars inside its alerts (the transaction view) misses that value.

Replay of the out-of-time window, one transaction at a time in record order:
- a transaction on a blocked card is declined (fraud $ prevented, or a legitimate
  purchase declined until the replacement card arrives);
- otherwise, a score at or above the cutoff raises an alert. If the day's analyst
  capacity is used up the alert goes unworked;
- a worked alert that turns out to be fraud blocks the card, effective per policy.
"""

import numpy as np
import pandas as pd

POLICIES = {
    # Analyst works the alert after the fact; block takes effect from the next day.
    "next_day_block": "block from the following day; the alerted fraud is lost",
    # Analyst works the alert the same day; later transactions that day are declined.
    "same_day_block": "block right after the alerted transaction; the alerted fraud is lost",
    # Every alert is declined in real time (legitimate ones too); fraud then blocks the card.
    "realtime_decline": "decline every alerted transaction; block the card if it was fraud",
}
WORKED = ["alert_fraud", "alert_legit", "declined_at_alert", "declined_false_alert"]
STOPPED = ["declined", "declined_at_alert"]


def replay(
    txns: pd.DataFrame,
    score: np.ndarray,
    cutoff: float,
    policy: str,
    capacity_per_day: int | None = None,
    reissue_lag_days: int | None = None,
) -> pd.DataFrame:
    """Return `txns` with a per-transaction `status` column for the given policy.

    capacity_per_day: alerts analysts can work per day (first come, first served); None = unlimited.
    reissue_lag_days: days until a blocked card's replacement arrives. Legitimate purchases on a
        blocked card are declined only within the lag; afterwards they happen on the new card.
        None = never replaced (every later legitimate purchase counts as declined).
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    t = txns[["Recnum", "Cardnum", "Date", "Amount", "Fraud"]].copy()
    t["score"] = score
    t = t.sort_values(["Date", "Recnum"])
    blocked_from: dict[str, tuple] = {}  # card -> (date, recnum) from which it is blocked
    worked_today: dict = {}
    lag = None if reissue_lag_days is None else pd.Timedelta(days=reissue_lag_days)
    status = []
    for rec, card, date, fraud, s in zip(t["Recnum"], t["Cardnum"], t["Date"], t["Fraud"], t["score"]):
        b = blocked_from.get(card)
        if b is not None and (date, rec) >= b:
            if fraud == 1 or lag is None or date < b[0] + lag:
                status.append("declined")
            else:
                status.append("passed_reissued")
            continue
        if s < cutoff:
            status.append("passed")
            continue
        if capacity_per_day is not None and worked_today.get(date, 0) >= capacity_per_day:
            status.append("alert_unworked")
            continue
        worked_today[date] = worked_today.get(date, 0) + 1
        if policy == "realtime_decline":
            if fraud == 1:
                blocked_from[card] = (date, rec)
                status.append("declined_at_alert")
            else:
                status.append("declined_false_alert")
        elif fraud == 1:
            blocked_from[card] = (date + pd.Timedelta(days=1), -1) if policy == "next_day_block" else (date, rec + 1)
            status.append("alert_fraud")
        else:
            status.append("alert_legit")
    t["status"] = status
    return t


def summarize(outcome: pd.DataFrame) -> dict:
    fraud = outcome["Fraud"] == 1
    s = outcome["status"]
    worked = s.isin(WORKED)
    stopped = s.isin(STOPPED)
    fraud_cards = outcome.loc[fraud, "Cardnum"].unique()
    detected = outcome.loc[fraud & s.isin(["alert_fraud", "declined_at_alert"]), "Cardnum"].unique()
    return {
        "transactions": len(outcome),
        "alerts": int(worked.sum()),
        "alert_rate": float(worked.mean()),
        "alerts_unworked": int((s == "alert_unworked").sum()),
        "false_alerts": int(s.isin(["alert_legit", "declined_false_alert"]).sum()),
        "fraud_txns": int(fraud.sum()),
        "fraud_dollars": float(outcome.loc[fraud, "Amount"].sum()),
        "fraud_dollars_in_alerts": float(outcome.loc[fraud & worked, "Amount"].sum()),
        "fraud_dollars_prevented": float(outcome.loc[fraud & stopped, "Amount"].sum()),
        "fraud_txns_prevented": int((fraud & stopped).sum()),
        "legit_txns_declined": int((~fraud & s.isin(["declined", "declined_false_alert"])).sum()),
        "fraud_cards": len(fraud_cards),
        "fraud_cards_detected": len(detected),
    }


def bootstrap_prevented_share(outcome: pd.DataFrame, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% interval for the share of fraud dollars prevented, resampling compromised cards.

    Fraud cards, not transactions, are the independent units here, and there are few
    of them in a two-month window, so the interval is wide on purpose.
    """
    f = outcome[outcome["Fraud"] == 1]
    stopped = f["status"].isin(STOPPED)
    per_card = pd.DataFrame({
        "fraud": f.groupby("Cardnum")["Amount"].sum(),
        "prevented": f[stopped].groupby("Cardnum")["Amount"].sum(),
    }).fillna(0.0)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(per_card), size=(n, len(per_card)))
    shares = per_card["prevented"].to_numpy()[pick].sum(1) / per_card["fraud"].to_numpy()[pick].sum(1)
    lo, hi = np.quantile(shares, [0.025, 0.975])
    return float(lo), float(hi)


def detection_delay(outcome: pd.DataFrame) -> pd.DataFrame:
    """Per fraud card: frauds and dollars that got through before the first true alert."""
    f = outcome[outcome["Fraud"] == 1]
    rows = []
    for card, g in f.groupby("Cardnum", sort=False):
        hit = g["status"].isin(["alert_fraud", "declined_at_alert"]).to_numpy()
        first = int(np.argmax(hit)) if hit.any() else len(g)
        rows.append({
            "Cardnum": card, "fraud_txns": len(g), "detected": bool(hit.any()),
            "frauds_before_first_alert": first,
            "dollars_before_first_alert": float(g["Amount"].iloc[:first].sum()),
            "fraud_dollars": float(g["Amount"].sum()),
        })
    return pd.DataFrame(rows)
