"""Phase 2 decision analysis: final model, cutoff, staffing, and value under uncertainty.

The cutoff is chosen on Sep-Oct (validation) with the model trained Jan 15 - Aug 31,
then applied unchanged to Nov-Dec (out-of-time). The out-of-time curve is also shown,
labeled as hindsight, to check the choice was not lucky.
"""

import json

import numpy as np
import pandas as pd

from . import config, economics, tuning, value
from .metrics import fdr_at
from .models import make_model
from .splits import periods

ALERT_RATES = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10)
CAPACITIES = (2, 4, 8, 16, None)  # alerts analysts can work per day
OOT_DAYS = 61


def choose_final(results: pd.DataFrame) -> dict:
    """Best mean time-fold test FDR over every evaluated config (OOT is never consulted)."""
    pick = results.loc[results["test"].idxmax()]
    family = pick["model"]
    float_params = {"learning_rate", "C"}
    params = {k: None if pd.isna(pick[k]) else (float(pick[k]) if k in float_params else int(pick[k]))
              for k in tuning.GRIDS[family]}
    return {"model": family, "params": params, "n_vars": int(pick["n_vars"]),
            "test": float(pick["test"]), "test_se": float(pick["test_se"]), "train": float(pick["train"]),
            "configs_within_one_se": int(tuning.within_one_se(results).sum()), "configs_evaluated": len(results)}


def fit_final(df, feats, rankings, final: dict):
    """Train the chosen config on fold B's training rows (Jan 15 - Aug 31)."""
    X, y = feats.drop(columns="Recnum"), df["Fraud"]
    _, train_b, test_b = list(tuning.fold_indices(df))[1]
    cols = rankings["B"][: final["n_vars"]]
    model = tuning._set_params(make_model(final["model"]), final["params"]).fit(X.loc[train_b, cols], y.loc[train_b])
    return model, cols, train_b, test_b


def analyze(df, feats, rankings, final: dict, a: economics.Assumptions = economics.Assumptions()) -> dict:
    out_dir = config.ARTIFACTS_DIR
    X = feats.drop(columns="Recnum")
    model, cols, train_b, val_idx = fit_final(df, feats, rankings, final)
    oot_idx = df.index[periods(df["Date"])[1]]
    s_val = model.predict_proba(X.loc[val_idx, cols])[:, 1]
    s_oot = model.predict_proba(X.loc[oot_idx, cols])[:, 1]
    val, oot = df.loc[val_idx], df.loc[oot_idx]

    perf = pd.DataFrame([
        {"window": w, "records": len(idx), "frauds": int(df.loc[idx, "Fraud"].sum()),
         **{f"fdr@{int(r * 100)}%": fdr_at(df.loc[idx, "Fraud"], model.predict_proba(X.loc[idx, cols])[:, 1], r)
            for r in (0.01, 0.02, 0.03, 0.05)}}
        for w, idx in (("train (Jan 15-Aug 31)", train_b), ("validation (Sep-Oct)", val_idx), ("out-of-time (Nov-Dec)", oot_idx))
    ])
    perf.to_csv(out_dir / "final_model_performance.csv", index=False)

    # 1. Unconstrained cutoff: maximize net value on validation.
    val_curve = economics.sweep(val, s_val, s_val, ALERT_RATES, a)
    oot_curve = economics.sweep(oot, s_oot, s_val, ALERT_RATES, a)  # same cutoffs, hindsight check
    val_curve.to_csv(out_dir / "value_curve_validation.csv", index=False)
    oot_curve.to_csv(out_dir / "value_curve_oot.csv", index=False)
    best = val_curve.loc[val_curve["net_value"].idxmax()]
    chosen = oot_curve.loc[oot_curve["target_alert_rate"] == best["target_alert_rate"]].iloc[0]

    # 2. Capacity-constrained: best cutoff for each daily review capacity.
    cap_rows = []
    for cap in CAPACITIES:
        c_val = economics.sweep(val, s_val, s_val, ALERT_RATES, a, capacity=cap)
        b = c_val.loc[c_val["net_value"].idxmax()]
        c_oot = economics.sweep(oot, s_oot, s_val, [b["target_alert_rate"]], a, capacity=cap).iloc[0]
        cap_rows.append({"capacity_per_day": cap if cap else "unlimited", "chosen_alert_rate": b["target_alert_rate"],
                         "val_net_value": b["net_value"], "oot_net_value": c_oot["net_value"],
                         "oot_prevented": c_oot["fraud_dollars_prevented"], "oot_unworked": c_oot["alerts_unworked"]})
    pd.DataFrame(cap_rows).to_csv(out_dir / "capacity_tradeoff.csv", index=False)

    cutoff = float(best["cutoff"])

    # Response speed: the same cutoff under each blocking policy, on both windows.
    policy_rows = []
    for window, w_txns, w_score in (("validation", val, s_val), ("oot", oot, s_oot)):
        for policy in value.POLICIES:
            s = value.summarize(value.replay(w_txns, w_score, cutoff, policy, None, a.reissue_lag_days))
            policy_rows.append({"window": window, "policy": policy, **s,
                                "prevented_share": s["fraud_dollars_prevented"] / s["fraud_dollars"],
                                **economics.net_value(s, a)})
    pd.DataFrame(policy_rows).to_csv(out_dir / "policy_comparison.csv", index=False)
    fraud = df[df["Fraud"] == 1]
    first_day = fraud["Date"] == fraud.groupby("Cardnum")["Date"].transform("min")
    first_day_share = float(fraud.loc[first_day, "Amount"].sum() / fraud["Amount"].sum())

    # 3. Uncertainty at the chosen cutoff.
    tor = economics.tornado(oot, s_oot, cutoff, a)
    tor.to_csv(out_dir / "tornado.csv", index=False)
    mc = economics.monte_carlo(oot, s_oot, cutoff, a)
    mc.to_csv(out_dir / "monte_carlo.csv", index=False)
    mc_cards = economics.monte_carlo(oot, s_oot, cutoff, a, vary_assumptions=False)

    # Simple rule baseline: the single strongest variable as the score, optimized the same way.
    rule_var = rankings["B"][0]
    r_val, r_oot = X.loc[val_idx, rule_var].to_numpy(), X.loc[oot_idx, rule_var].to_numpy()
    rule_curve = economics.sweep(val, r_val, r_val, ALERT_RATES, a)
    r_best = rule_curve.loc[rule_curve["net_value"].idxmax()]
    rule_oot = economics.sweep(oot, r_oot, r_val, [r_best["target_alert_rate"]], a).iloc[0]

    annual = 365 / OOT_DAYS
    headline = {
        "assumptions": a.__dict__,
        "final_model": final,
        "chosen_alert_rate": float(best["target_alert_rate"]),
        "cutoff": cutoff,
        "oot": {k: float(chosen[k]) for k in (
            "alerts", "alert_rate", "false_alerts", "fraud_dollars", "fraud_dollars_prevented",
            "legit_txns_declined", "fraud_cards", "fraud_cards_detected", "benefit", "cost_total", "net_value")},
        "oot_best_alert_rate_hindsight": float(oot_curve.loc[oot_curve["net_value"].idxmax(), "target_alert_rate"]),
        "staffing": economics.staffing(int(chosen["alerts"]), OOT_DAYS, a),
        "net_value_annualized_run_rate": float(chosen["net_value"] * annual),
        "monte_carlo_oot_net_value": {q: float(mc["net_value"].quantile(p)) for q, p in
                                      (("p05", 0.05), ("p50", 0.5), ("p95", 0.95))},
        "monte_carlo_prob_negative": float((mc["net_value"] < 0).mean()),
        "monte_carlo_cards_only": {q: float(mc_cards["net_value"].quantile(p)) for q, p in
                                   (("p05", 0.05), ("p50", 0.5), ("p95", 0.95))},
        "fraud_dollars_on_first_fraud_day_share": first_day_share,
        "policy_comparison": [{k: r[k] for k in ("window", "policy", "fraud_dollars_prevented", "prevented_share",
                                                 "alerts", "legit_txns_declined", "net_value")} for r in policy_rows],
        "rule_baseline": {"variable": rule_var, "alert_rate": float(r_best["target_alert_rate"]),
                          "oot_net_value": float(rule_oot["net_value"]),
                          "oot_prevented": float(rule_oot["fraud_dollars_prevented"]),
                          "oot_alerts": int(rule_oot["alerts"])},
    }
    (out_dir / "decision.json").write_text(json.dumps(headline, indent=2, default=float))
    return headline
