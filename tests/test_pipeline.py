import numpy as np
import pandas as pd
import pytest

from fraudscore import config, data, features, splits, value
from fraudscore.metrics import fdr_at, ks_statistic


def txns(rows):
    """Build a small transaction frame from (recnum, card, date, merch, amount, fraud) tuples."""
    df = pd.DataFrame(rows, columns=["Recnum", "Cardnum", "Date", "Merchnum", "Amount", "Fraud"])
    df["Date"] = pd.to_datetime(df["Date"])
    df["Merch_desc"] = "DESC " + df["Merchnum"]
    df["Merch_state"] = "CA"
    df["Merch_zip"] = "90095"
    return df


# --- metrics -------------------------------------------------------------

def test_fdr_known_answer():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 1] * 10)  # 100 records, 30 frauds
    score = np.where(y == 1, 1.0, 0.0)
    score[:3] = [0.5, 2.0, 2.0]  # two goods outrank every fraud
    # top 3% = 3 records: two goods + one fraud -> 1/30 frauds found
    assert fdr_at(y, score, 0.03) == pytest.approx(1 / 30)
    assert fdr_at(y, np.where(y == 1, 1.0, 0.0), 0.30) == 1.0


def test_psi_zero_for_same_distribution_and_grows_with_shift():
    from fraudscore.metrics import psi
    rng = np.random.default_rng(0)
    base = rng.normal(size=20_000)
    assert psi(base, rng.normal(size=20_000)) < 0.01
    assert psi(base, rng.normal(loc=1.0, size=20_000)) > 0.25


def test_ks_separates_and_ignores_ties():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic([1, 2, 3, 4], y) == 1.0
    assert ks_statistic([1, 1, 1, 1], y) == 0.0


# --- cleaning ------------------------------------------------------------

def test_exclusions_keep_purchases_and_drop_amount_errors():
    raw = pd.DataFrame({"Transtype": ["P", "A", "P", "D"], "Amount": [10, 5, 3_000_000, 7], "Fraud": [0, 0, 0, 0]})
    kept, log = data.apply_exclusions(raw)
    assert kept["Amount"].tolist() == [10]
    assert log["records"].tolist() == [2, 1]


def test_cleaning_imputation_order():
    df = pd.DataFrame({
        "Merchnum": ["111", "111", np.nan, "0", np.nan],
        "Merch_desc": ["ACME", "ACME", "ACME", "NEWCO", "ONLYONCE"],
        "Merch_state": ["NY", "NY", np.nan, np.nan, "TX"],
        "Merch_zip": [10001.0, 10001.0, np.nan, 2134.0, np.nan],
        "Fraud": 0,
    })
    zips = pd.DataFrame({"zip": ["10001", "02134", "75001", "75002"], "state": ["NY", "MA", "TX", "TX"],
                         "irs_estimated_population": [5, 5, 10, 99]})
    out, _ = data.clean(df, zips)
    assert out.loc[2, "Merchnum"] == "111"  # filled from same description
    assert out.loc[3, "Merchnum"] == "D:NEWCO"  # '0' is a placeholder -> new id from description
    assert out.loc[3, "Merch_state"] == "MA"  # looked up from its ZIP
    assert out.loc[2, "Merch_state"] == "NY" and out.loc[2, "Merch_zip"] == "10001"  # from merchant
    assert out.loc[4, "Merch_zip"] == "75002"  # most populated ZIP in TX
    assert out["Merch_zip"].str.len().eq(5).all()


def test_non_us_state_codes_become_foreign():
    df = pd.DataFrame({"Merchnum": ["1"], "Merch_desc": ["X"], "Merch_state": ["297"], "Merch_zip": [np.nan], "Fraud": 0})
    zips = pd.DataFrame({"zip": ["10001"], "state": ["NY"], "irs_estimated_population": [1]})
    out, _ = data.clean(df, zips)
    assert out.loc[0, "Merch_state"] == data.FOREIGN


# --- features: no look-ahead ---------------------------------------------

def test_window_counts_are_point_in_time():
    df = txns([
        (1, "A", "2010-01-01", "m1", 10, 0),
        (2, "A", "2010-01-01", "m2", 20, 0),
        (3, "A", "2010-01-05", "m1", 30, 0),
        (4, "B", "2010-01-05", "m1", 40, 0),
        (5, "A", "2010-01-20", "m1", 50, 0),
    ])
    f = features.build_features(df).set_index("Recnum")
    assert f.loc[1, "card_count_0"] == 1 and f.loc[2, "card_count_0"] == 2  # same day: earlier records only
    assert f.loc[3, "card_count_7"] == 3 and f.loc[3, "card_total_7"] == 60
    assert f.loc[5, "card_count_14"] == 1  # Jan 5 is 15 days back, outside the 14-day window
    assert f.loc[3, "card_days_since"] == 4
    assert f.loc[3, "card_max_7"] == 30
    assert f.loc[4, "merch_count_30"] == 3  # merchant m1 across cards


def test_future_records_do_not_change_past_features():
    rows = [(i, "A" if i % 2 else "B", f"2010-01-{1 + i // 3:02d}", f"m{i % 3}", 10.0 * i, 0) for i in range(1, 30)]
    full = features.build_features(txns(rows)).set_index("Recnum")
    early = features.build_features(txns(rows[:15])).set_index("Recnum")
    pd.testing.assert_frame_equal(full.loc[early.index], early)


# --- splits --------------------------------------------------------------

def _dev_frame(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(rng.choice(pd.date_range("2010-01-01", "2010-12-31"), n))
    return pd.DataFrame({
        "Date": dates, "Cardnum": rng.integers(0, 200, n).astype(str), "Fraud": (rng.random(n) < 0.1).astype(int),
    })


def test_oot_window_is_disjoint_and_after_development():
    df = _dev_frame()
    dev, oot = splits.periods(df["Date"])
    assert not (dev & oot).any()
    assert df.loc[dev, "Date"].max() < config.OOT_START <= df.loc[oot, "Date"].min()
    assert df.loc[dev, "Date"].min() >= config.MODEL_START


def test_card_split_never_shares_a_card():
    df = _dev_frame()
    dev = df[splits.periods(df["Date"])[0]]
    train, test = splits.card_split(dev, seed=1)
    assert set(dev.loc[train, "Cardnum"]).isdisjoint(dev.loc[test, "Cardnum"])
    assert 0.2 < len(test) / len(dev) < 0.4


def test_temporal_split_respects_time():
    df = _dev_frame()
    dev = df[splits.periods(df["Date"])[0]]
    train, test = splits.temporal_split(dev)
    assert dev.loc[train, "Date"].max() < dev.loc[test, "Date"].min()


# --- card-stop value -----------------------------------------------------

def _burst():
    return txns([
        (1, "A", "2010-11-01", "m", 100, 1),
        (2, "A", "2010-11-01", "m", 200, 1),
        (3, "A", "2010-11-02", "m", 300, 1),
        (4, "B", "2010-11-02", "m", 50, 0),
    ])


@pytest.mark.parametrize("policy, prevented", [
    ("next_day_block", 300),       # alert on #1, block from Nov 2
    ("same_day_block", 500),       # #2 and #3 declined
    ("realtime_decline", 600),     # #1 declined too
])
def test_replay_policies(policy, prevented):
    t = _burst()
    score = np.array([0.9, 0.1, 0.1, 0.95])
    out = value.replay(t, score, cutoff=0.5, policy=policy)
    s = value.summarize(out)
    assert s["fraud_dollars_prevented"] == prevented
    assert s["false_alerts"] == 1  # card B's alert
    assert s["fraud_cards_detected"] == 1


def test_realtime_declines_legit_alerts_too():
    out = value.replay(_burst(), np.array([0.9, 0.1, 0.1, 0.95]), 0.5, "realtime_decline")
    assert out.set_index("Recnum").loc[4, "status"] == "declined_false_alert"
    assert value.summarize(out)["legit_txns_declined"] == 1


def test_daily_capacity_leaves_extra_alerts_unworked():
    t = _burst()
    out = value.replay(t, np.array([0.1, 0.1, 0.9, 0.95]), 0.5, "next_day_block", capacity_per_day=1)
    st = out.set_index("Recnum")["status"]
    assert st[3] == "alert_fraud" and st[4] == "alert_unworked"  # first come, first served on Nov 2


def test_reissue_lag_limits_declined_purchases():
    t = txns([
        (1, "A", "2010-11-01", "m", 100, 1),
        (2, "A", "2010-11-03", "m", 20, 0),   # inside a 5-day lag: declined
        (3, "A", "2010-11-10", "m", 30, 0),   # after the replacement card: goes through
        (4, "A", "2010-11-11", "m", 400, 1),  # fraudster still has the old number: declined
    ])
    out = value.replay(t, np.array([0.9, 0, 0, 0]), 0.5, "next_day_block", reissue_lag_days=5)
    assert out["status"].tolist() == ["alert_fraud", "declined", "passed_reissued", "declined"]


def test_net_value_arithmetic():
    from fraudscore.economics import Assumptions, net_value
    s = {"fraud_dollars_prevented": 1000, "alerts": 12, "fraud_cards_detected": 2,
         "false_alerts": 10, "legit_txns_declined": 3}
    a = Assumptions(minutes_per_review=10, analyst_cost_per_hour=60, loss_share=0.5,
                    reissue_cost=20, false_alert_cost=1, declined_purchase_cost=5)
    # 500 benefit - (12*10 review + 40 reissue + 10 false alerts + 15 declines)
    assert net_value(s, a)["net_value"] == pytest.approx(500 - 185)


def test_detection_delay_counts_frauds_before_first_alert():
    out = value.replay(_burst(), np.array([0.1, 0.9, 0.1, 0.0]), 0.5, "next_day_block")
    d = value.detection_delay(out).set_index("Cardnum")
    assert d.loc["A", "frauds_before_first_alert"] == 1
    assert d.loc["A", "dollars_before_first_alert"] == 100


# --- reason codes and briefs -------------------------------------------------

def test_variable_descriptions_and_families():
    from fraudscore.explain import describe, family
    assert describe("card_total_3") == "Card: spend, 3-day window"
    assert describe("card_count_0") == "Card: purchases, same day"
    assert describe("card_merch_days_since") == "Card at this merchant: days since previous purchase"
    assert describe("merch_velocity_1_by_30") == "Merchant: purchases in the last 2 days vs. its 30-day pace"
    assert family("card_total_1") == family("card_total_30") != family("card_count_1")
    assert family("card_merch_total_7") != family("card_total_7")


def test_reason_codes_find_the_variable_that_drives_the_score():
    from sklearn.linear_model import LogisticRegression
    from fraudscore.explain import Explainer
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"card_total_3": rng.normal(size=2000), "merch_count_7": rng.normal(size=2000)})
    y = pd.Series((X["card_total_3"] + 0.1 * rng.normal(size=2000) > 1.5).astype(int))
    model = LogisticRegression().fit(X, y)
    ex = Explainer(model, list(X.columns), X, y)
    alert = pd.DataFrame({"card_total_3": [3.0], "merch_count_7": [0.0]})
    reasons = ex.reasons(alert)[0]
    assert reasons[0]["variable"] == "card_total_3"
    assert reasons[0]["percentile"] > 0.99


def test_grounding_rejects_invented_numbers():
    from fraudscore.explain import grounded
    brief = "Card spend $4,210 (typical $120). 9 purchases in 7 days."
    assert grounded("Spend of $4,210 is far above the typical $120 over 7 days.", brief)
    assert not grounded("Spend of $4,210 is 35 times the norm.", brief)


class _FakeClient:
    """Stands in for anthropic.Anthropic(): returns a fixed reply, records the request."""

    def __init__(self, text, stop_reason="end_turn"):
        block = type("Block", (), {"type": "text", "text": text})()
        self.reply = type("Msg", (), {"content": [block], "stop_reason": stop_reason})()
        self.beta = self
        self.messages = self
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


@pytest.mark.parametrize("reply, stop, source", [
    ("Card spend of $4,210 vs typical $120. Confirm today.", "end_turn", "llm"),
    ("Spend is 35 times normal. Block now.", "end_turn", "template"),
    ("", "refusal", "template"),
])
def test_llm_summary_kept_only_when_grounded(reply, stop, source):
    from fraudscore.explain import narrate
    b = {"text": "A-001 · Card spend $4,210 (typical $120). Confirm with the cardholder today."}
    client = _FakeClient(reply, stop)
    out = narrate(b, client)
    assert out["narrative_source"] == source
    assert out["narrative"] == (reply if source == "llm" else b["text"])
    assert client.calls[0]["model"] == "claude-opus-5-5" and client.calls[0]["fallbacks"] == "default"
