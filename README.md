# NFL 2026 Season Prediction Model

## Project Overview

This project uses machine learning to predict each NFL team's win total for the 2026 season.
Three regression models are trained and compared: Linear Regression (with feature scaling),
Random Forest, and Gradient Boosting. The best model is automatically selected by 5-fold
cross-validation MAE and used for final predictions.

The model combines three input signals:

1. **Past performance** — 2025 season wins, scoring stats, offensive/defensive rankings
2. **Draft quality** — 2026 NFL Draft pick scores aggregated by team
3. **Roster movement** — Off-season trade and free-agency scores

---

## Data Pipeline

```
draft_text.txt
    └── clean_draft_data.py  ──>  data/clean_nfl_draft_data.csv
                                         │
data/team_stats.csv  ──────────────────> merge (left join on team abbreviation)
data/trades.csv  ───────────────────────>│
                                         │
                                    main.py
                                         │
                              engineer_features()
                                         │
                              train_models()  ──>  3 models (LR, RF, GB)
                                         │
                              evaluate_models()  ──>  MAE / RMSE / R² / 5-fold CV
                                         │
                              generate_predictions()
                                         │
                     ┌───────────────────┴───────────────────────┐
              outputs/predictions_2026.csv           outputs/*.png (4 charts)
```

**Key design decisions:**
- `clean_draft_data.py` is called automatically if the draft CSV is missing.
- A left-merge on `team_stats.csv` ensures all 32 teams appear even if draft or trade data is absent; missing values are filled with 0.
- Final predictions retrain the chosen model on all 32 teams (no held-out set) to maximize signal before generating 2026 forecasts.

---

## Folder Structure

```
NFL Prediction/
├── clean_draft_data.py          # Parses draft_text.txt  ->  data/clean_nfl_draft_data.csv
├── draft_text.txt               # Raw 2026 NFL Draft picks (text format)
├── main.py                      # Full pipeline — run this to train models and predict
│
├── data/
│   ├── clean_nfl_draft_data.csv # Cleaned draft picks (auto-generated)
│   ├── team_stats.csv           # 2025 team performance stats
│   └── trades.csv               # Off-season trade/FA scores
│
├── outputs/
│   ├── predictions_2026.csv     # Predicted wins for all 32 teams
│   ├── predicted_wins_chart.png # Horizontal bar chart (red -> green)
│   ├── feature_importance.png   # Ranked feature importance from best model
│   ├── draft_vs_wins.png        # Scatter: draft score vs predicted wins
│   └── wins_comparison.png      # Scatter: 2025 wins vs 2026 predicted wins
│
└── README.md
```

---

## How to Run

**Step 1 — Install required packages (first time only):**

```
pip install pandas numpy scikit-learn matplotlib
```

**Step 2 — Run the main script:**

```
python main.py
```

The script will automatically:
- Create `data/team_stats.csv` and `data/trades.csv` with sample data if they don't exist
- Run `clean_draft_data.py` to generate the draft CSV from `draft_text.txt` if needed
- Train all three models and print a comparison table
- Save `outputs/predictions_2026.csv` and four charts

To regenerate the draft CSV from scratch:

```
python clean_draft_data.py
```

---

## Feature Engineering

### Raw features (from `team_stats.csv`)

| Feature | Description |
|---------|-------------|
| `wins_last` | Wins in the 2025 season |
| `offensive_rank` | Offensive ranking (1 = best) |
| `defensive_rank` | Defensive ranking (1 = best) |
| `strength_of_schedule` | Average opponent win percentage |

### Derived performance features (new in v2)

| Feature | Formula | Description |
|---------|---------|-------------|
| `win_pct_last` | wins / (wins + losses) | Prior season win percentage |
| `points_per_game` | points_for / 17 | Offensive scoring rate |
| `points_allowed_per_game` | points_against / 17 | Defensive yield rate |
| `net_points_per_game` | point_diff / 17 | Net scoring margin per game |
| `elite_team_last_year` | 1 if wins >= 11 | Flag: likely playoff contender |
| `rebuild_team` | 1 if wins <= 6 | Flag: team in rebuild mode |

### Draft features (from `clean_nfl_draft_data.csv`)

| Feature | Description |
|---------|-------------|
| `total_draft_score` | Sum of all pick scores |
| `num_picks` | Total draft picks made |
| `first_round_picks` | Number of first-round picks |
| `offensive_picks` | Picks at QB/RB/WR/TE/OT/G/C/FB |
| `defensive_picks` | Picks at DE/DT/LB/CB/S/etc. |
| `qb_drafted` | 1 if a QB was drafted |
| `draft_efficiency` | total_draft_score / num_picks |
| `early_pick_ratio` | first_round_picks / num_picks |

### Roster movement features (from `trades.csv`)

| Feature | Description |
|---------|-------------|
| `trade_score` | Net off-season roster change score |
| `roster_change_index` | total_draft_score + trade_score |

**Total: 20 features.**

### Draft pick scoring formula

Each pick is scored as: **round value + position bonus**

| Round | Base Score | Position | Bonus |
|-------|-----------|----------|-------|
| 1st | 10 | QB | +4 |
| 2nd | 7  | WR, DE, OLB, CB, OT | +3 |
| 3rd | 5  | DT, DL, LB, S, TE | +2 |
| 4th | 3  | RB, G, OG, C | +1 |
| 5th–7th | 2 or 1 | All others | 0 |

---

## Model Comparison

Three models are trained on a 75/25 train-test split (24 training teams, 8 test teams).
Model selection uses 5-fold cross-validation MAE — more reliable than a single small test
split when n=32.

| Model | Train MAE | Test MAE | Test RMSE | Test R² | 5-fold CV MAE |
|-------|-----------|----------|-----------|---------|--------------|
| Linear Regression | 0.13 | 0.38 | 0.46 | 0.903 | 0.51 |
| **Random Forest** | **0.20** | **0.41** | **0.52** | **0.878** | **0.50** |
| Gradient Boosting | 0.00 | 0.58 | 0.72 | 0.760 | 0.53 |

**Selected: Random Forest** (lowest 5-fold CV MAE = 0.50)

Linear Regression achieves slightly better test metrics but Random Forest generalizes better
across all 5 folds. Gradient Boosting overfits (Train MAE ≈ 0 vs Test MAE = 0.58).

### Top feature importances (Random Forest)

| Feature | Importance |
|---------|-----------|
| net_points_per_game | 0.313 |
| win_pct_last | 0.127 |
| points_per_game | 0.125 |
| offensive_rank | 0.119 |
| wins_last | 0.093 |

---

## Key Insights

**Most improved teams (2025 → 2026 predicted):**
- NE: +1.1 wins — strong off-season trade score (+3) and high draft investment
- ARI: +0.9 wins — low baseline (4W) makes regression to the mean likely
- NYG: +0.7 wins — draft and trade activity signal intent to compete

**Most declining teams (2025 → 2026 predicted):**
- KC: -1.6 wins — mean reversion from 14W; still projected #1 at 12.4W
- PHI: -1.3 wins — projected to drop from 13W to 11.7W
- DAL: -1.1 wins — negative prior season trajectory continues

**Draft capital leaders:**
- MIA holds the highest total draft score (87) — projected 9.3W, biggest potential upside
- CLE (72) and LV (66) also invested heavily but have poor prior records (6W, 5W)

**Correlation analysis:**
- Prior season wins has by far the strongest correlation with predicted wins (r = +0.99)
- Draft score adds modest but positive signal (r = +0.08)
- With only one season of training data, the model is heavily anchored to `wins_last`

---

## Limitations

- **Small dataset** — Only 32 teams. With a 75/25 split, 8 teams are used for testing.
  R² is unstable at this sample size and should not be over-interpreted.
- **Sample data** — `team_stats.csv` and `trades.csv` contain fictional values.
  Results improve substantially with real multi-year historical data.
- **One season of training data** — Real NFL prediction models use 5–10+ seasons.
  With n=32 and one target year, the model essentially learns mean-reversion.
- **Draft text parsing** — The regex parser handles 254/254 picks from the 2026 draft
  sim correctly, but unusual name formats or future draft text may require adjustment.
- **No in-season data** — Injuries, coaching changes, and mid-season trades are not captured.
- **Overfitting risk** — Gradient Boosting achieves near-zero train error (0.00 MAE),
  confirming it memorizes the small training set rather than generalizing.

---

## Future Improvements

- **Multi-year training data** — Stack 5+ years of season stats to dramatically increase
  n and reduce reliance on mean-reversion as the dominant signal.
- **XGBoost / LightGBM** — Install `xgboost` for a fourth model comparison; it is
  already wired into `main.py` and activates automatically when available.
- **Quarterback-specific features** — Add starter QBR, passer rating, or injury flag,
  which are strong independent predictors of win totals.
- **Coaching and front-office changes** — Encode head coach tenure, off-season staff
  turnover, and GM changes as binary features.
- **Advanced draft metrics** — Replace the manual scoring formula with publicly available
  draft-pick value charts (e.g., Jimmy Johnson chart, ESPN draft grades).
- **Real trade data** — Replace the sample `trade_score` column with actual snap counts,
  contract values, or EPA-based player impact estimates for arriving/departing players.
- **Ensemble stacking** — Use Linear Regression predictions as a meta-feature on top of
  tree model outputs to combine the strengths of each model type.

---

## File Descriptions

| File | Role |
|------|------|
| `clean_draft_data.py` | Reads `draft_text.txt`, extracts picks with regex, scores each pick, saves CSV |
| `main.py` | Orchestrates everything: data loading, feature engineering, training, evaluation, output |
| `draft_text.txt` | Raw input — 2026 NFL Draft picks in unstructured concatenated text format |
| `data/team_stats.csv` | Input — team performance from 2025 season (sample data) |
| `data/trades.csv` | Input — off-season roster change scores (sample data) |
| `outputs/predictions_2026.csv` | Output — ranked table of 2026 predicted win totals |
| `outputs/predicted_wins_chart.png` | Output — bar chart, red (few wins) to green (many wins) |
| `outputs/feature_importance.png` | Output — ranked feature importances from the best model |
| `outputs/draft_vs_wins.png` | Output — scatter: draft score vs predicted wins with trend line |
| `outputs/wins_comparison.png` | Output — scatter: 2025 actual wins vs 2026 predicted wins |
