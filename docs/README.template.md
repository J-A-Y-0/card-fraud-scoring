<h1 align="center">Stopping the card</h1>

<p align="center"><b>Card-fraud scoring judged by the fraud it stops</b><br>
The model finds the run. A confirmed alert blocks the card. Speed decides the value.</p>

<p align="center">
  <a href="https://github.com/J-A-Y-0/card-fraud-scoring/actions/workflows/ci.yml"><img src="https://github.com/J-A-Y-0/card-fraud-scoring/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/scikit--learn-1.7-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn 1.7">
  <img src="https://img.shields.io/badge/fraud%20%24%20stopped-{same_share_badge}%20same%20day-1167a8" alt="Fraud dollars stopped {same_share}">
</p>

![Demo: switching response speed in the planner, then working an alert in the queue](docs/img/demo.gif)

<p align="center"><sub>The planner compares response speeds on two months the model never saw. The alert queue shows why each alert fired; the analyst makes the call.</sub></p>

<p align="center"><a href="https://j-a-y-0.github.io/card-fraud-scoring/"><b>Planner</b></a> · <a href="https://j-a-y-0.github.io/card-fraud-scoring/alerts.html"><b>Alert queue</b></a> · <a href="reports/fraud-scoring-report.pdf"><b>Report (PDF)</b></a> · <a href="reports/fraud-scoring-deck.pdf"><b>Deck (PDF)</b></a></p>

## The business case

**The problem.** A purchase-card program loses money to compromised cards. Analysts can check only a few alerts a day, and the usual model test overstates how much fraud a score catches.

**What I found**, on Nov–Dec 2010, months the model never saw:

- **The usual test is wrong.** A random train/test split reported {rand_test} detection. On later months, the same model caught {rand_oot}.
- **The value is in the card.** {first_day} of fraud dollars land on a card's first fraud day. A confirmed alert is worth the rest of the run it stops.
- **Speed beats coverage.** With the same alerts, blocking the card the same day stops more fraud for less work.

| Out-of-time, Nov–Dec 2010 | Block next day | Block same day |
| --- | --- | --- |
| Fraud dollars stopped | {next_share} | **{same_share}** |
| Net value, two months | {net} | **{same_net}** |
| Analyst time | {next_minutes} min a day | **{same_minutes} min a day** |

**Recommendation.** Review the top 3% of scores ({oot_fdr} of fraud found), block cards the same day, and run a 60-day shadow pilot before going live.

## Who does what

| Job | Who |
| --- | --- |
| Rank every purchase by fraud risk | **Model** |
| Explain each alert: reasons and measured fraud rate | **Code** |
| Call the cardholder, confirm or clear | **Analyst** |
| Block and reissue the card | **Card operations** |
| Set the review budget and response speed | **Program owner**, with the planner |

## How it works

```mermaid
flowchart LR
    A["Purchases<br/>98,161"] --> B["Point-in-time variables<br/>earlier records only"]
    B --> C["Tune on time-ordered folds"]
    C --> D["Replay Nov–Dec<br/>block the card on a confirmed alert"]
    D --> E["Net value, staffing,<br/>Monte Carlo"]
    D --> F["Alert briefs<br/>reasons + fraud rate"]
```

| Validation | Response speed |
| --- | --- |
| ![Random-split test vs. out-of-time detection](docs/img/validation.png) | ![Share of fraud dollars stopped by response speed](docs/img/speed.png) |
| A random split overstates detection by {rand_gap} points; a time-ordered split lands within {temp_gap}. | Blocking sooner stops more of each run. |

## Quick start

Runs in about two minutes on synthetic data. No dataset or API key needed.

```sh
git clone https://github.com/J-A-Y-0/card-fraud-scoring.git
cd card-fraud-scoring
uv sync
uv run fraudscore synth
FRAUD_DATA_DIR=data/synthetic FRAUD_ARTIFACTS_DIR=artifacts/synthetic uv run fraudscore all --quick
uv run pytest
```

To browse the tools locally: `python -m http.server -d site`, then open <http://localhost:8000>.

## Project structure

| | |
| --- | --- |
| `src/fraudscore/` | cleaning, variables, splits, tuning, card-block replay, economics, alert briefs |
| `site/` | planner, alert queue, report and deck (static HTML) |
| `reports/` | report and deck PDFs |
| `tests/` | unit tests, offline |

## Limitations

- **Small sample.** {fraud_cards_oot} compromised cards in the test window: the direction of each finding is solid, its size is uncertain.
- **Assumptions.** Labels are taken as complete, and a blocked card is assumed to stop the fraudster. Costs are planning values you can change in the planner.
- **Data.** The transaction file is not redistributed. `site/data/` holds only aggregates and a relabeled, rounded alert sample.
