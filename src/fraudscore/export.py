"""Export aggregated, publishable JSON for the report, deck and decision tool.

Nothing here is transaction-level: cards are renumbered, amounts are summed per
card-day, and no merchant, ZIP or card-number field leaves the pipeline.
"""

import json

import numpy as np
import pandas as pd

from . import config, data, decision, economics, explain, features, splits, value

SITE_DATA = config.ROOT / "site" / "data"
SAMPLE_START, SAMPLE_END = pd.Timestamp("2010-12-06"), pd.Timestamp("2010-12-19")  # published alert-queue sample
TOOL_RATES = tuple(np.round(np.r_[np.arange(0.0025, 0.03, 0.0025), np.arange(0.03, 0.1001, 0.01)], 4))


def _site_dir():
    """Real-data runs publish to site/data; any other dataset (e.g. synthetic) stays in artifacts."""
    real = config.DATA_DIR.resolve() == (config.ROOT / "data" / "raw").resolve()
    return SITE_DATA if real else config.ARTIFACTS_DIR / "site-data"


def _write(name, obj):
    out = _site_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(obj, indent=1, default=float))


def burst_timeline(outcome: pd.DataFrame) -> list[dict]:
    """Per compromised card: daily lost / stopped fraud dollars and the first-alert day."""
    start = outcome["Date"].min()
    fraud_cards = outcome.loc[outcome["Fraud"] == 1, "Cardnum"].unique()
    o = outcome[outcome["Cardnum"].isin(fraud_cards)].copy()
    o["day"] = (o["Date"] - start).dt.days
    stopped = o["status"].isin(value.STOPPED)
    o["lost"] = np.where((o["Fraud"] == 1) & ~stopped, o["Amount"], 0.0)
    o["stopped"] = np.where((o["Fraud"] == 1) & stopped, o["Amount"], 0.0)
    o["legit"] = (o["Fraud"] == 0).astype(int)
    o["alert"] = o["status"].isin(["alert_fraud", "declined_at_alert"])
    order = o.groupby("Cardnum")[["lost", "stopped"]].sum().sum(1).sort_values(ascending=False).index
    cards = []
    for i, card in enumerate(order, 1):
        g = o[o["Cardnum"] == card]
        days = g.groupby("day").agg(lost=("lost", "sum"), stopped=("stopped", "sum"), legit=("legit", "sum")).reset_index()
        alert_days = g.loc[g["alert"], "day"]
        fraud_days = g.loc[g["Fraud"] == 1, "day"]
        cards.append({
            "card": f"Card {i:02d}", "first_fraud_day": int(fraud_days.min()),
            "first_alert_day": int(alert_days.min()) if len(alert_days) else None,
            "lost": float(g["lost"].sum()), "stopped": float(g["stopped"].sum()),
            "days": days[(days["lost"] > 0) | (days["stopped"] > 0) | (days["legit"] > 0)].to_dict("records"),
        })
    return cards


def run() -> None:
    art = config.ARTIFACTS_DIR
    df, _, _ = data.load_clean()
    feats = features.build_features(df)
    rankings = pd.read_pickle(art / "rankings.pkl")
    final = json.loads((art / "final_model.json").read_text())
    dec = json.loads((art / "decision.json").read_text())
    a = economics.Assumptions()

    model, cols, train_b, val_idx = decision.fit_final(df, feats, rankings, final)
    X = feats.drop(columns="Recnum")
    oot_idx = df.index[splits.periods(df["Date"])[1]]
    s_val = model.predict_proba(X.loc[val_idx, cols])[:, 1]
    s_oot = model.predict_proba(X.loc[oot_idx, cols])[:, 1]
    oot = df.loc[oot_idx]

    gap = pd.read_csv(art / "split_gap_summary.csv")
    _write("split_gap.json", gap.to_dict("records"))
    _write("policy.json", pd.read_csv(art / "policy_comparison.csv").to_dict("records"))
    perf = pd.read_csv(art / "final_model_performance.csv").set_index("window")
    _write("headline.json", {**dec, "oot_days": decision.OOT_DAYS, "oot_start": str(config.OOT_START.date()),
                             "performance": perf.reset_index().to_dict("records"),
                             "oot_fdr_at_3": float(perf.loc["out-of-time (Nov-Dec)", "fdr@3%"])})

    outcome = value.replay(oot, s_oot, dec["cutoff"], "next_day_block", None, a.reissue_lag_days)
    _write("bursts.json", {"start": str(oot["Date"].min().date()), "days": decision.OOT_DAYS,
                           "policy": "next_day_block", "cards": burst_timeline(outcome)})

    # Decision tool: summaries for every (policy, alert rate, reissue lag); money is applied in the browser.
    grid = []
    for policy in value.POLICIES:
        for rate in TOOL_RATES:
            cutoff = float(np.quantile(s_val, 1 - rate))
            for lag in range(economics.RANGES["reissue_lag_days"][0], economics.RANGES["reissue_lag_days"][1] + 1):
                s = value.summarize(value.replay(oot, s_oot, cutoff, policy, None, lag))
                grid.append({"policy": policy, "rate": float(rate), "lag": lag, **{k: s[k] for k in (
                    "alerts", "false_alerts", "fraud_dollars", "fraud_dollars_prevented",
                    "legit_txns_declined", "fraud_cards", "fraud_cards_detected", "transactions")}})
    _write("tool_grid.json", {"days": decision.OOT_DAYS, "defaults": a.__dict__, "ranges": economics.RANGES,
                              "productive_hours_per_analyst_day": economics.PRODUCTIVE_HOURS_PER_ANALYST_DAY,
                              "rows": grid})
    # Analyst briefs for the recommended same-day policy. Published sample: two weeks,
    # cards relabeled, dollar amounts rounded to $10.
    same_day = value.replay(oot, s_oot, dec["cutoff"], "same_day_block", None, a.reissue_lag_days)
    val_fraud = df.loc[val_idx, "Fraud"].to_numpy()
    labels: dict[str, str] = {}
    label = lambda c: labels.setdefault(c, f"C-{len(labels) + 1:02d}")
    public = explain.Explainer(model, cols, X.loc[train_b], df.loc[train_b, "Fraud"], money_unit=10)
    window = same_day[(same_day["Date"] >= SAMPLE_START) & (same_day["Date"] <= SAMPLE_END)]
    sample = explain.alert_briefs(df, feats, window, public, s_val, val_fraud, card_label=label)
    full = explain.alert_briefs(df, feats, same_day, explain.Explainer(model, cols, X.loc[train_b], df.loc[train_b, "Fraud"]),
                                s_val, val_fraud)
    _write("alerts.json", {"start": str(SAMPLE_START.date()), "end": str(SAMPLE_END.date()), "policy": "same_day_block",
                           "bands": explain.score_bands(s_val, val_fraud), "alerts": sample,
                           "reason_mix": explain.reason_mix(full).to_dict("records")})

    numbers = report_numbers(df, feats, rankings, final, model, cols, s_oot, outcome, a)
    numbers["alerts"] = {"total": len(full), "true": sum(b["outcome"]["fraud"] for b in full),
                         "bands": explain.score_bands(s_val, val_fraud),
                         "reason_mix": explain.reason_mix(full).to_dict("records"),
                         "example": next(b for b in sample if b["outcome"]["fraud"])}
    _write("report.json", numbers)
    if _site_dir() == SITE_DATA:
        render_readme(numbers, dec, perf, pd.read_csv(art / "policy_comparison.csv"))
    print(f"wrote {sorted(p.name for p in _site_dir().glob('*.json'))} to {_site_dir()}")


def render_readme(R: dict, dec: dict, perf: pd.DataFrame, policy: pd.DataFrame) -> None:
    """Fill docs/README.template.md with this run's numbers so the README cannot drift."""
    pct = lambda v, d=0: f"{v * 100:.{d}f}%"
    usd_k = lambda v: f"${v / 1000:,.0f}k"
    pol = policy[policy["window"] == "oot"].set_index("policy")
    gap = {r["split"]: r for r in R["split_gap"] if r["model"] == "boosted_trees"}
    rf = {r["split"]: r for r in R["split_gap"] if r["model"] == "random_forest"}
    days = decision.OOT_DAYS
    per_day = pol.loc["next_day_block", "alerts"] / days
    minutes = per_day * R["assumptions"]["minutes_per_review"]
    values = {
        "oot_fdr": pct(perf.loc["out-of-time (Nov-Dec)", "fdr@3%"], 1), "val_fdr": pct(perf.loc["validation (Sep-Oct)", "fdr@3%"], 1),
        "next_share": pct(pol.loc["next_day_block", "prevented_share"]), "same_share": pct(pol.loc["same_day_block", "prevented_share"]),
        "rt_share": pct(pol.loc["realtime_decline", "prevented_share"]),
        "next_per_day": f"{per_day:.1f}", "next_minutes": f"{minutes:.0f}",
        "same_net": f"${pol.loc['same_day_block', 'net_value']:,.0f}",
        "same_minutes": f"{pol.loc['same_day_block', 'alerts'] / days * R['assumptions']['minutes_per_review']:.0f}",
        "next_fte": f"{minutes / 60 / economics.PRODUCTIVE_HOURS_PER_ANALYST_DAY:.2f}",
        "net": f"${dec['oot']['net_value']:,.0f}", "mc_p05": usd_k(R["monte_carlo"]["p05"]), "mc_p95": usd_k(R["monte_carlo"]["p95"]),
        "model_vs_rule": f"{dec['oot']['net_value'] / dec['rule_baseline']['oot_net_value']:.1f}×",
        "rule_net": usd_k(dec["rule_baseline"]["oot_net_value"]),
        "rand_gap": f"{(gap['random']['test'] - gap['random']['oot']) * 100:.0f}",
        "temp_gap": f"{(gap['temporal']['test'] - gap['temporal']['oot']) * 100:.0f}",
        "leak": pct(gap["random"]["test_frauds_on_train_fraud_cards"]),
        "fraud_cards": R["data"]["fraud_cards"], "fraud_records": f"{R['data']['fraud_records']:,}",
        "first_day": pct(R["data"]["first_day_dollar_share"]),
        "cost_share": pct(dec["oot"]["cost_total"] / dec["oot"]["benefit"]),
        "candidates": R["variables"]["candidates"], "configs": R["tuning"]["configs"], "mc_draws": f"{R['monte_carlo']['draws']:,}",
        "fraud_cards_oot": R["pilot"]["oot_fraud_cards"],
        "rand_test": pct(gap["random"]["test"]), "rand_oot": pct(gap["random"]["oot"]),
        "rf_rand_gap": f"{(rf['random']['test'] - rf['random']['oot']) * 100:.0f}",
        "band_urgent": pct(R["alerts"]["bands"][0]["fraud_rate"]), "band_review": pct(R["alerts"]["bands"][1]["fraud_rate"]),
        "report_pages": (config.ROOT / "site" / "report.html").read_text().count('<article class="page'),
    }
    values["oot_fdr_badge"] = values["oot_fdr"].replace("%", "%25")
    values["same_share_badge"] = values["same_share"].replace("%", "%25")
    template = (config.ROOT / "docs" / "README.template.md").read_text()
    (config.ROOT / "README.md").write_text(template.format(**values))


def data_quality(raw: pd.DataFrame) -> dict:
    """Field-level profile of the raw file (the DQR), aggregated only."""
    fields = []
    for col in raw.columns:
        s = raw[col]
        numeric = col in ("Recnum", "Amount", "Fraud")
        row = {"field": col, "type": "numeric" if numeric else ("date" if col == "Date" else "categorical"),
               "populated": float(s.notna().mean()), "unique": int(s.nunique()),
               "zeros": int((s == 0).sum()) if numeric else None}
        if numeric:
            row.update(min=float(s.min()), max=float(s.max()), mean=float(s.mean()), median=float(s.median()))
        elif col == "Date":
            row.update(min=str(s.min().date()), max=str(s.max().date()))
        else:
            row["most_common_share"] = float(s.value_counts(normalize=True).iloc[0])
        fields.append(row)
    month = raw.groupby(raw["Date"].dt.month).agg(records=("Fraud", "size"), fraud=("Fraud", "sum"))
    month["fraud_rate"] = month["fraud"] / month["records"]
    edges = np.logspace(-2, 7, 37)
    hist = {}
    for label, part in (("legit", raw[raw["Fraud"] == 0]), ("fraud", raw[raw["Fraud"] == 1])):
        counts, _ = np.histogram(part["Amount"].clip(lower=0.01), bins=edges)
        hist[label] = (counts / counts.sum()).tolist()
    return {
        "records": len(raw), "fields": fields, "transtype": raw["Transtype"].value_counts().to_dict(),
        "months": month.reset_index().rename(columns={"Date": "month"}).to_dict("records"),
        "amount_hist": {"edges": edges.tolist(), **hist},
        "states_top10": raw["Merch state"].value_counts().head(10).to_dict() if "Merch state" in raw else
        raw["Merch_state"].value_counts().head(10).to_dict(),
    }


def report_numbers(df, feats, rankings, final, model, cols, s_oot, outcome, a) -> dict:
    """Every computed quantity the report prose refers to."""
    from .metrics import ks_statistic, psi
    from .tuning import fdr_card_se, fold_indices

    art = config.ARTIFACTS_DIR
    raw = data.load_raw()
    fraud = df[df["Fraud"] == 1]
    per_card = fraud.groupby("Cardnum")["Amount"].agg(["size", "sum"]).sort_values("sum", ascending=False)
    first_day = fraud["Date"] == fraud.groupby("Cardnum")["Date"].transform("min")
    frauds_per_card = per_card["size"].value_counts().sort_index()

    X = feats.drop(columns="Recnum")
    _, train_b, _ = list(fold_indices(df))[1]
    oot_idx = df.index[splits.periods(df["Date"])[1]]
    final_vars = [{"variable": c, "ks": ks_statistic(X.loc[train_b, c], df.loc[train_b, "Fraud"]),
                   "psi_oot": psi(X.loc[train_b, c], X.loc[oot_idx, c])} for c in cols]
    score_psi = psi(model.predict_proba(X.loc[train_b, cols])[:, 1], s_oot)

    se16 = fdr_card_se(df.loc[oot_idx, "Fraud"], s_oot, df.loc[oot_idx, "Cardnum"])
    n_cards = int(fraud.loc[fraud.index.isin(oot_idx), "Cardnum"].nunique())
    delay = value.detection_delay(outcome)
    mc = pd.read_csv(art / "monte_carlo.csv")["net_value"]
    mc_edges = np.linspace(mc.min(), mc.max(), 31)
    tuning_res = pd.read_csv(art / "tuning_results.csv")
    grid = tuning_res[tuning_res["stage"] == "grid"]
    best_by_family = grid.loc[grid.groupby("model")["test"].idxmax()]

    return {
        "data": {
            "raw_records": len(raw), "clean_records": len(df), "cards": int(df["Cardnum"].nunique()),
            "merchants": int(df["Merchnum"].nunique()), "fraud_records": int(df["Fraud"].sum()),
            "fraud_rate": float(df["Fraud"].mean()), "fraud_dollars": float(fraud["Amount"].sum()),
            "fraud_cards": len(per_card), "frauds_per_card_median": float(per_card["size"].median()),
            "single_fraud_cards": int((per_card["size"] == 1).sum()),
            "top10_card_dollar_share": float(per_card["sum"].head(10).sum() / per_card["sum"].sum()),
            "first_day_dollar_share": float(fraud.loc[first_day, "Amount"].sum() / fraud["Amount"].sum()),
            "frauds_per_card_hist": {int(k): int(v) for k, v in frauds_per_card.items()},
        },
        "dqr": data_quality(raw),
        "exclusions": pd.read_csv(art / "exclusions.csv").to_dict("records"),
        "imputation": pd.read_csv(art / "imputation_log.csv").to_dict("records"),
        "variables": {"candidates": int(X.shape[1]), "entities": list(features.ENTITIES),
                      "windows": list(features.WINDOWS), "final": final_vars},
        "split_gap": pd.read_csv(art / "split_gap_summary.csv").to_dict("records"),
        "tuning": {
            "configs": len(tuning_res), "within_one_se": final["configs_within_one_se"],
            "n_vars": tuning_res[tuning_res["stage"] == "n_vars"][["n_vars", "train", "test", "test_se", "oot"]].to_dict("records"),
            "best_by_family": best_by_family[["model", "train", "test", "test_se", "oot"]].to_dict("records"),
            "final": final,
        },
        "complexity": pd.read_csv(art / "complexity_curves.csv").to_dict("records"),
        "performance": pd.read_csv(art / "final_model_performance.csv").to_dict("records"),
        "value_curve": {"validation": pd.read_csv(art / "value_curve_validation.csv")[["target_alert_rate", "net_value", "alerts", "fraud_dollars_prevented"]].to_dict("records"),
                        "oot": pd.read_csv(art / "value_curve_oot.csv")[["target_alert_rate", "net_value", "alerts", "fraud_dollars_prevented"]].to_dict("records")},
        "capacity": pd.read_csv(art / "capacity_tradeoff.csv").to_dict("records"),
        "tornado": pd.read_csv(art / "tornado.csv").to_dict("records"),
        "monte_carlo": {"p05": float(mc.quantile(0.05)), "p50": float(mc.quantile(0.5)), "p95": float(mc.quantile(0.95)),
                        "prob_negative": float((mc < 0).mean()), "draws": len(mc),
                        "hist": np.histogram(mc, bins=mc_edges)[0].tolist(), "edges": mc_edges.tolist()},
        "delay": {"median_frauds": float(delay["frauds_before_first_alert"].median()),
                  "median_dollars": float(delay["dollars_before_first_alert"].median()),
                  "total_dollars": float(delay["dollars_before_first_alert"].sum()),
                  "caught_on_first_fraud": int((delay["frauds_before_first_alert"] == 0).sum()),
                  "detected": int(delay["detected"].sum()), "cards": len(delay)},
        "monitoring": {"score_psi_oot": score_psi},
        "pilot": {"oot_fraud_cards": n_cards, "fdr_se_at_oot": se16,
                  "se_by_cards": {n: se16 * np.sqrt(n_cards / n) for n in (16, 32, 48, 96)}},
        "assumptions": a.__dict__, "ranges": economics.RANGES,
    }
