"""Command-line entry point.

    fraudscore all              run -> tune -> decide -> briefs -> export
    fraudscore synth            write a synthetic dataset to data/synthetic/
    FRAUD_DATA_DIR=data/synthetic fraudscore all --quick     the whole pipeline in about a minute
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from . import config, data, decision, economics, experiments, features, splits, tuning, value


def run(seeds) -> None:
    """Clean, build variables, and audit how much each split strategy flatters the model."""
    out = config.ARTIFACTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    print("cleaning")
    df, excl, imp = data.load_clean()
    excl.to_csv(out / "exclusions.csv", index=False)
    imp.to_csv(out / "imputation_log.csv", index=False)
    print("building point-in-time variables")
    feats = features.build_features(df)
    print("validation audit: split strategies x seeds")
    runs = experiments.run_split_gap(df, feats, seeds)
    runs.to_csv(out / "split_gap_runs.csv", index=False)
    summary = experiments.summarize_split_gap(runs)
    summary.to_csv(out / "split_gap_summary.csv", index=False)
    print(summary.to_string(index=False))


def _prepared():
    df, _, _ = data.load_clean()
    return df, features.build_features(df)


def tune(quick: bool) -> None:
    df, feats = _prepared()
    results, rankings = tuning.search(df, feats, quick=quick)
    results.to_csv(config.ARTIFACTS_DIR / "tuning_results.csv", index=False)
    pd.to_pickle(rankings, config.ARTIFACTS_DIR / "rankings.pkl")
    final = decision.choose_final(results)
    (config.ARTIFACTS_DIR / "final_model.json").write_text(json.dumps(final, indent=2))
    base = {k: final["params"][k] for k in tuning.GRIDS["boosted_trees"]} if final["model"] == "boosted_trees" \
        else {"learning_rate": 0.1, "max_iter": 100, "max_leaf_nodes": 15, "min_samples_leaf": 200}
    leaves = [3, 15, 63] if quick else [3, 7, 15, 31, 63, 127]
    min_leaf = [400, 50] if quick else [800, 400, 200, 100, 50, 20, 5]
    curves = pd.concat([
        tuning.complexity_curve(df, feats, rankings, final["n_vars"], base, "max_leaf_nodes", leaves),
        tuning.complexity_curve(df, feats, rankings, final["n_vars"], base, "min_samples_leaf", min_leaf),
    ])
    curves.to_csv(config.ARTIFACTS_DIR / "complexity_curves.csv", index=False)
    print(json.dumps(final, indent=2))


def _load_final():
    rankings = pd.read_pickle(config.ARTIFACTS_DIR / "rankings.pkl")
    final = json.loads((config.ARTIFACTS_DIR / "final_model.json").read_text())
    return rankings, final


def decide() -> None:
    df, feats = _prepared()
    rankings, final = _load_final()
    print(json.dumps(decision.analyze(df, feats, rankings, final), indent=2, default=float))


def briefs(llm: bool, limit: int) -> None:
    """Analyst briefs for every Nov-Dec alert under same-day blocking (local, full precision)."""
    from . import explain
    df, feats = _prepared()
    rankings, final = _load_final()
    cutoff = json.loads((config.ARTIFACTS_DIR / "decision.json").read_text())["cutoff"]
    model, cols, train_idx, val_idx = decision.fit_final(df, feats, rankings, final)
    X = feats.drop(columns="Recnum")
    oot_idx = df.index[splits.periods(df["Date"])[1]]
    s_val = model.predict_proba(X.loc[val_idx, cols])[:, 1]
    s_oot = model.predict_proba(X.loc[oot_idx, cols])[:, 1]
    outcome = value.replay(df.loc[oot_idx], s_oot, cutoff, "same_day_block", None,
                           economics.Assumptions().reissue_lag_days)
    ex = explain.Explainer(model, cols, X.loc[train_idx], df.loc[train_idx, "Fraud"])
    out = explain.alert_briefs(df, feats, outcome, ex, s_val, df.loc[val_idx, "Fraud"].to_numpy())
    if llm:
        import anthropic  # optional dependency: uv sync --extra llm
        client = anthropic.Anthropic()
        out = [explain.narrate(b, client) if i < limit else b for i, b in enumerate(out)]
        kept = sum(b.get("narrative_source") == "llm" for b in out)
        print(f"LLM summaries kept for {kept} of {min(limit, len(out))} briefs; the rest use the template")
    (config.ARTIFACTS_DIR / "briefs.json").write_text(explain.dumps(out))
    print(f"{len(out)} briefs -> {config.ARTIFACTS_DIR / 'briefs.json'}\n")
    print(out[0].get("narrative", out[0]["text"]))


def export() -> None:
    from . import export as ex
    ex.run()


def main() -> None:
    parser = argparse.ArgumentParser(prog="fraudscore", description="Card fraud scoring: validation, value and decision")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="clean, build variables, validation audit")
    p.add_argument("--seeds", type=int, default=len(config.SEEDS), help="repeats per split strategy")
    t = sub.add_parser("tune", help="time-fold model search, final model, complexity curves")
    t.add_argument("--quick", action="store_true", help="small grid, for smoke tests")
    sub.add_parser("decide", help="cutoff, staffing, response policy and value under uncertainty")
    b = sub.add_parser("briefs", help="reason codes and analyst briefs for every out-of-time alert")
    b.add_argument("--llm", action="store_true", help="add a Claude-written summary, kept only if fully grounded")
    b.add_argument("--limit", type=int, default=20, help="max briefs sent to the LLM (cost control)")
    sub.add_parser("export", help="aggregated JSON for the site, report and deck, and README.md")
    a = sub.add_parser("all", help="run, tune, decide, briefs and export in order")
    a.add_argument("--quick", action="store_true", help="1 seed and a small grid")
    s = sub.add_parser("synth", help="write a synthetic dataset with the real file's schema")
    s.add_argument("--out", type=Path, default=config.ROOT / "data" / "synthetic")
    args = parser.parse_args()

    if args.command == "run":
        run(tuple(range(args.seeds)))
    elif args.command == "tune":
        tune(args.quick)
    elif args.command == "decide":
        decide()
    elif args.command == "briefs":
        briefs(args.llm, args.limit)
    elif args.command == "export":
        export()
    elif args.command == "all":
        run(tuple(range(1 if args.quick else len(config.SEEDS))))
        tune(args.quick)
        decide()
        briefs(False, 0)
        export()
    elif args.command == "synth":
        from . import synthetic
        print(f"wrote {synthetic.write(args.out)}")
