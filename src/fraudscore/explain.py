"""Reason codes and analyst briefs for alerts.

Reason codes are measured, not guessed: each variable in turn is reset to its typical
value (the median of non-fraud training purchases) and the drop in the score is that
variable's contribution to this alert. Related windows of the same statistic are grouped
so an alert does not list "spend over 1 day" and "spend over 3 days" as two reasons.

A brief is a deterministic summary an analyst can act on in under a minute. An optional
LLM pass may rewrite it as prose, but only if every number it uses comes from the brief.
"""

import json
import re

import numpy as np
import pandas as pd

ENTITY_NAMES = {
    "card": "Card", "merch": "Merchant", "card_merch": "Card at this merchant",
    "card_zip": "Card in this ZIP", "card_state": "Card in this state", "merch_zip": "Merchant in this ZIP",
}
STAT_NAMES = {
    "count": "purchases", "total": "spend", "avg": "average purchase", "max": "largest purchase",
    "actual_over_avg": "this purchase vs. its average", "days_since": "days since previous purchase",
    "velocity": "burst of activity",
}
FIRST_SEEN = 365  # features.build_features uses 365 days for "no earlier purchase"


def split_name(var: str) -> tuple[str, str, str]:
    """'card_total_3' -> ('card', 'total', '3'); 'card_velocity_1_by_30' -> ('card', 'velocity', '1_by_30')."""
    if var in ("amount", "amount_cents_zero"):
        return var, var, ""
    entity = max((e for e in ENTITY_NAMES if var.startswith(e + "_")), key=len)
    rest = var[len(entity) + 1:]
    for stat in sorted(STAT_NAMES, key=len, reverse=True):
        if rest == stat or rest.startswith(stat + "_"):
            return entity, stat, rest[len(stat) + 1:]
    raise ValueError(f"unrecognized variable {var!r}")


def describe(var: str) -> str:
    """Plain-language name for a variable, e.g. 'Card spend, 3-day window'."""
    if var == "amount":
        return "Purchase amount"
    if var == "amount_cents_zero":
        return "Round-dollar amount"
    entity, stat, window = split_name(var)
    if stat == "days_since":
        return f"{ENTITY_NAMES[entity]}: days since previous purchase"
    if stat == "velocity":
        short, long = window.split("_by_")
        when = "today" if short == "0" else "in the last 2 days"
        return f"{ENTITY_NAMES[entity]}: purchases {when} vs. its {long}-day pace"
    span = "same day" if window == "0" else f"{window}-day window"
    return f"{ENTITY_NAMES[entity]}: {STAT_NAMES[stat]}, {span}"


def family(var: str) -> str:
    entity, stat, _ = split_name(var)
    return f"{entity}:{stat}"


def money(v: float, unit: int = 1) -> str:
    """Dollar text, optionally rounded to `unit` dollars (used for published samples)."""
    return f"${round(v / unit) * unit:,.0f}"


def fmt_value(var: str, v: float, unit: int = 1) -> str:
    _, stat, _ = split_name(var)
    if var == "amount_cents_zero":
        return "yes" if v >= 0.5 else "no"
    if stat in ("count",):
        return f"{v:,.0f}"
    if stat == "days_since":
        return "first purchase" if v >= FIRST_SEEN else f"{v:,.0f} days"
    if stat in ("actual_over_avg", "velocity"):
        return f"{v:,.1f}×"
    return money(v, unit)


class Explainer:
    """Reason codes for a fitted model, relative to its training population."""

    def __init__(self, model, cols: list[str], X_train: pd.DataFrame, y_train: pd.Series, money_unit: int = 1):
        self.model, self.cols, self.money_unit = model, cols, money_unit
        good = X_train.loc[y_train == 0, cols]
        self.typical = good.median()
        self.sorted_train = {c: np.sort(X_train[c].to_numpy()) for c in cols}

    def percentile(self, var: str, v: float) -> float:
        s = self.sorted_train[var]
        return float(np.searchsorted(s, v, side="right") / len(s))

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Score drop when each variable is reset to its typical value (rows = alerts, cols = variables)."""
        Xa = X[self.cols]
        base = self.model.predict_proba(Xa)[:, 1]
        n, k = len(Xa), len(self.cols)
        stacked = pd.DataFrame(np.repeat(Xa.to_numpy(), k, axis=0), columns=self.cols)
        for j, c in enumerate(self.cols):
            stacked.iloc[j::k, j] = self.typical[c]
        reset = self.model.predict_proba(stacked)[:, 1].reshape(n, k)
        return pd.DataFrame(base[:, None] - reset, index=X.index, columns=self.cols)

    def reasons(self, X: pd.DataFrame, top: int = 3) -> list[list[dict]]:
        contrib = self.contributions(X)
        out = []
        for idx, row in contrib.iterrows():
            best: dict[str, tuple[str, float]] = {}
            for var, c in row.items():
                f = family(var)
                if c > 0 and (f not in best or c > best[f][1]):
                    best[f] = (var, c)
            ranked = sorted(best.values(), key=lambda t: -t[1])[:top]
            # Drop minor reasons: each listed reason must explain at least 15% as much as the first.
            ranked = [(v, c) for v, c in ranked if c >= 0.15 * ranked[0][1]] if ranked else []
            out.append([{
                "variable": var, "label": describe(var), "contribution": float(c),
                "value": float(X.at[idx, var]), "typical": float(self.typical[var]),
                "value_text": fmt_value(var, X.at[idx, var], self.money_unit),
                "typical_text": fmt_value(var, self.typical[var], self.money_unit),
                "first_seen": split_name(var)[1] == "days_since" and X.at[idx, var] >= FIRST_SEEN,
                "percentile": self.percentile(var, X.at[idx, var]),
            } for var, c in ranked])
        return out


def brief(alert: dict, reasons: list[dict], context: dict, band: dict, unit: int = 1) -> dict:
    """Deterministic analyst brief.

    alert: id, date, card, amount, score, score_pct. band: name and the share of validation
    purchases in this score band that were fraud (a calibrated, measured likelihood).
    """
    lines = []
    for i, r in enumerate(reasons, 1):
        if r.get("first_seen"):
            rank = "no earlier purchase"
        elif r["percentile"] >= 0.5:
            rank = f"higher than {r['percentile']:.0%} of purchases"
        else:
            rank = f"lower than {1 - r['percentile']:.0%} of purchases"
        lines.append(f"{i}. {r['label']}: {r['value_text']} (typical {r['typical_text']}; {rank}).")
    ctx = [f"{context['card_count_7']:,.0f} purchases on this card in the last 7 days, {money(context['card_total_7'], unit)} in total"]
    ctx.append("first purchase at this merchant" if context["new_merchant"] else "card has bought here before")
    urgent = band["name"] == "urgent"
    action = ("Call the cardholder now; if the purchase is not recognized, block the card at once."
              if urgent else "Confirm with the cardholder today; if the purchase is not recognized, block the card.")
    text = (f"{alert['id']} · {alert['date']} · {alert['card']} · {money(alert['amount'], unit)}\n"
            f"Score in the top {alert['score_pct']:.1%} of purchases. In this band, "
            f"{band['fraud_rate']:.0%} of Sep–Oct alerts were fraud.\nWhy:\n" + "\n".join(lines) +
            f"\nContext: {'; '.join(ctx)}.\nAction: {action}")
    return {**alert, "priority": band["name"], "band_fraud_rate": band["fraud_rate"], "reasons": reasons,
            "context": context, "action": action, "text": text}


def score_bands(val_scores: np.ndarray, val_fraud: np.ndarray, urgent_rate: float = 0.01, alert_rate: float = 0.03) -> list[dict]:
    """Two alert bands with their measured fraud rate on validation purchases."""
    order = np.argsort(-val_scores)
    k1, k3 = int(len(order) * urgent_rate), int(len(order) * alert_rate)
    s = val_scores[order]
    return [
        {"name": "urgent", "min_score": float(s[k1 - 1]), "fraud_rate": float(val_fraud[order[:k1]].mean())},
        {"name": "review", "min_score": float(s[k3 - 1]), "fraud_rate": float(val_fraud[order[k1:k3]].mean())},
    ]


# ---- optional LLM write-up ---------------------------------------------------

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

SYSTEM = (
    "You write alert summaries for card-fraud analysts. Rewrite the brief you are given as two or "
    "three short sentences: what is unusual, how unusual, and the action. Use only facts and numbers "
    "that appear in the brief, written exactly as they appear there. Do not add advice, causes or "
    "probabilities that the brief does not state."
)


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in NUMBER.findall(text)}


def grounded(narrative: str, brief_text: str) -> bool:
    """True if every number in the narrative also appears in the brief."""
    return _numbers(narrative) <= _numbers(brief_text)


def narrate(b: dict, client, model: str = "claude-opus-5-5") -> dict:
    """Ask Claude for a prose version of a brief; keep it only if it is grounded.

    `client` is an anthropic.Anthropic() (or a test double with the same surface).
    Falls back to the deterministic text on refusal, empty output or any ungrounded number.
    """
    response = client.beta.messages.create(
        model=model,
        max_tokens=1024,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low"},
        system=SYSTEM,
        messages=[{"role": "user", "content": b["text"]}],
    )
    text = "" if response.stop_reason == "refusal" else "".join(
        block.text for block in response.content if block.type == "text").strip()
    ok = bool(text) and grounded(text, b["text"])
    return {**b, "narrative": text if ok else b["text"], "narrative_source": "llm" if ok else "template"}


def alert_briefs(df: pd.DataFrame, feats: pd.DataFrame, outcome: pd.DataFrame, explainer: Explainer,
                 val_scores: np.ndarray, val_fraud: np.ndarray, card_label=str) -> list[dict]:
    """Briefs for every worked alert in a replay outcome (from fraudscore.value.replay).

    card_label maps a card number to the label shown; pass a renumbering for anything published.
    """
    worked = outcome[outcome["status"].isin(["alert_fraud", "alert_legit", "declined_at_alert", "declined_false_alert"])]
    idx = df.index[df["Recnum"].isin(worked["Recnum"])]
    X = feats.loc[idx].drop(columns="Recnum")
    reasons = explainer.reasons(X)
    status = worked.set_index("Recnum")["status"]
    ref = np.sort(val_scores)
    bands = score_bands(val_scores, val_fraud)
    unit = explainer.money_unit
    out = []
    for i, (row_idx, rs) in enumerate(zip(idx, reasons), 1):
        rec = df.loc[row_idx]
        s = float(outcome.loc[outcome["Recnum"] == rec["Recnum"], "score"].iloc[0])
        alert = {"id": f"A-{i:03d}", "date": str(rec["Date"].date()), "card": card_label(rec["Cardnum"]),
                 "amount": float(round(rec["Amount"] / unit) * unit), "score": s,
                 "score_pct": float(max(1 - np.searchsorted(ref, s) / len(ref), 1 / len(ref)))}
        ctx = {"card_count_7": float(feats.at[row_idx, "card_count_7"]), "card_total_7": float(feats.at[row_idx, "card_total_7"]),
               "new_merchant": bool(feats.at[row_idx, "card_merch_days_since"] >= FIRST_SEEN)}
        band = bands[0] if s >= bands[0]["min_score"] else bands[1]
        if unit > 1:
            ctx["card_total_7"] = float(round(ctx["card_total_7"] / unit) * unit)
        b = brief(alert, rs, ctx, band, unit)
        b["outcome"] = {"fraud": bool(rec["Fraud"] == 1), "status": status[rec["Recnum"]]}
        out.append(b)
    return out


def reason_mix(briefs: list[dict]) -> pd.DataFrame:
    """How often each reason family leads an alert, for true vs false alerts."""
    rows = [{"top_reason": b["reasons"][0]["label"].split(",")[0].replace(" vs. its", " vs.") if b["reasons"] else "none",
             "fraud": b["outcome"]["fraud"]} for b in briefs]
    t = pd.DataFrame(rows).groupby(["top_reason", "fraud"]).size().unstack(fill_value=0)
    t.columns = ["false_alerts" if not c else "true_alerts" for c in t.columns]
    return t.reset_index().sort_values("true_alerts", ascending=False)


def dumps(obj) -> str:
    return json.dumps(obj, indent=1, default=float)
