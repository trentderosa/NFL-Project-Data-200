# =============================================================================
# NFL 2026 Season Prediction Model  |  DATA 200 Project  (v2 - Enhanced)
# Run:  python main.py
#
# Pipeline
#   1. load_data()           - load & merge all data sources
#   2. engineer_features()   - add derived features
#   3. train_models()        - train Linear Regression, RF, Gradient Boosting
#   4. evaluate_models()     - compare with MAE, RMSE, R2, CV
#   5. generate_predictions()- predict 2026 wins for all 32 teams
#   6. print_insights()      - most improved, declining, draft leaders
#   7. visualize_results()   - 4 charts saved to outputs/
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# =============================================================================
# PATHS & GLOBAL CONSTANTS
# =============================================================================

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

TEAM_NAME_MAP = {
    "Arizona Cardinals":    "ARI",  "Atlanta Falcons":      "ATL",
    "Baltimore Ravens":     "BAL",  "Buffalo Bills":        "BUF",
    "Carolina Panthers":    "CAR",  "Chicago Bears":        "CHI",
    "Cincinnati Bengals":   "CIN",  "Cleveland Browns":     "CLE",
    "Dallas Cowboys":       "DAL",  "Denver Broncos":       "DEN",
    "Detroit Lions":        "DET",  "Green Bay Packers":    "GB",
    "Houston Texans":       "HOU",  "Indianapolis Colts":   "IND",
    "Jacksonville Jaguars": "JAX",  "Kansas City Chiefs":   "KC",
    "Los Angeles Chargers": "LAC",  "Los Angeles Rams":     "LAR",
    "Las Vegas Raiders":    "LV",   "Miami Dolphins":       "MIA",
    "Minnesota Vikings":    "MIN",  "New England Patriots": "NE",
    "New Orleans Saints":   "NO",   "New York Giants":      "NYG",
    "New York Jets":        "NYJ",  "Philadelphia Eagles":  "PHI",
    "Pittsburgh Steelers":  "PIT",  "Seattle Seahawks":     "SEA",
    "San Francisco 49ers":  "SF",   "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans":     "TEN",  "Washington Commanders":"WAS",
}

OFFENSIVE_POSITIONS = {"QB", "RB", "WR", "TE", "OT", "OG", "G", "C", "FB"}
DEFENSIVE_POSITIONS = {"DE", "DT", "DL", "NT", "OLB", "LB", "CB", "S", "DB"}

TARGET = "wins_next"   # column in team_stats.csv used as training label


# =============================================================================
# SAMPLE DATA CREATION
# These blocks only run if the CSVs do not already exist.
# =============================================================================

TEAM_STATS_FILE = DATA_DIR / "team_stats.csv"

if not TEAM_STATS_FILE.exists():
    print("Creating sample team_stats.csv ...")

    # -------------------------------------------------------------------------
    # wins_next is the TRAINING LABEL only — it represents how many games the
    # team won the FOLLOWING season.  Replace with real historical data for
    # better accuracy (e.g. use 2024 wins when training on 2023 stats).
    # -------------------------------------------------------------------------
    stats = {
        "team": [
            "KC","PHI","DET","BAL","BUF","DAL","SF","HOU",
            "MIA","MIN","GB","LAR","PIT","ATL","SEA","TB",
            "WAS","NYJ","CHI","CIN","IND","NO","TEN","NYG",
            "JAX","LV","ARI","CAR","NE","DEN","LAC","CLE",
        ],
        "wins_last":   [14,13,12,12,12,11,11,11,10,10,10, 9, 9, 9, 8, 8, 8, 7, 7, 7, 7, 6, 6, 5, 5, 5, 4, 4, 4, 7, 8, 6],
        "losses_last": [ 3, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9,10,10,10,10,11,11,12,12,12,13,13,13,10, 9,11],
        "points_for":  [437,425,419,398,407,382,375,390,368,355,362,345,322,356,340,328,335,302,318,325,310,290,295,275,282,278,262,255,268,315,338,288],
        "points_against":[285,302,318,301,312,324,308,335,342,338,345,348,340,361,358,352,360,365,378,372,368,382,388,398,401,408,415,420,410,370,355,385],
        "point_diff":  [152,123,101, 97, 95, 58, 67, 55, 26, 17, 17, -3,-18, -5,-18,-24,-25,-63,-60,-47,-58,-92,-93,-123,-119,-130,-153,-165,-142,-55,-17,-97],
        "offensive_rank":[2,3,5,7,4,9,8,6,11,15,13,16,22,12,18,21,17,25,23,19,24,27,26,29,28,30,31,32,28,20,14,26],
        "defensive_rank":[5,8,11,9,12,14,10,16,18,20,19,21,17,23,22,15,24,20,25,26,22,27,28,29,30,31,32,32,31,23,24,28],
        "strength_of_schedule":[0.52,0.49,0.51,0.53,0.50,0.48,0.54,0.50,0.49,0.48,0.51,0.50,0.49,0.48,0.51,0.49,0.50,0.48,0.47,0.51,0.49,0.50,0.48,0.51,0.50,0.47,0.49,0.48,0.51,0.50,0.51,0.49],
        "wins_next":   [13,12,11,12,11,10,10,10, 9, 9,10, 9, 8, 8, 8, 8, 7, 7, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 5, 7, 8, 6],
    }
    pd.DataFrame(stats).to_csv(TEAM_STATS_FILE, index=False)
    print(f"  Saved: {TEAM_STATS_FILE}\n")


TRADES_FILE = DATA_DIR / "trades.csv"

if not TRADES_FILE.exists():
    print("Creating sample trades.csv ...")
    trades = {
        "team": ["KC","PHI","DET","BAL","BUF","DAL","SF","HOU","MIA","MIN","GB","LAR","PIT","ATL","SEA","TB","WAS","NYJ","CHI","CIN","IND","NO","TEN","NYG","JAX","LV","ARI","CAR","NE","DEN","LAC","CLE"],
        "trade_score": [2,1,-1,3,1,0,-2,2,1,-1,0,1,2,-1,0,1,2,3,1,-2,0,-1,2,1,-1,2,1,0,3,1,2,1],
    }
    pd.DataFrame(trades).to_csv(TRADES_FILE, index=False)
    print(f"  Saved: {TRADES_FILE}\n")


# =============================================================================
# HELPER
# =============================================================================

def _save_fig(fig, filename):
    """Save a matplotlib figure to outputs/ and close it."""
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# STEP 1 — load_data
# =============================================================================

def load_data():
    """
    Load draft picks, team stats, and trades. Aggregate draft to team level.
    Merge all three sources into one DataFrame. Return the merged DataFrame.
    """
    draft_file = DATA_DIR / "clean_nfl_draft_data.csv"
    if not draft_file.exists():
        print("Draft CSV not found. Running clean_draft_data.py ...")
        import clean_draft_data
        clean_draft_data.main()

    print("Loading data ...")
    df_draft  = pd.read_csv(draft_file)
    df_stats  = pd.read_csv(TEAM_STATS_FILE)
    df_trades = pd.read_csv(TRADES_FILE)
    print(f"  Draft picks : {len(df_draft):>4} rows")
    print(f"  Team stats  : {len(df_stats):>4} rows")
    print(f"  Trades      : {len(df_trades):>4} rows")

    # Map full team names to abbreviations
    df_draft["team_abbr"] = df_draft["team"].map(TEAM_NAME_MAP)
    unmapped = df_draft[df_draft["team_abbr"].isna()]["team"].unique()
    if len(unmapped):
        print(f"  Warning: unmapped team name(s): {unmapped}")
    df_draft = df_draft[df_draft["team_abbr"].notna()].copy()

    # Per-pick flags used in aggregation
    df_draft["is_offensive"]   = df_draft["position"].isin(OFFENSIVE_POSITIONS).astype(int)
    df_draft["is_defensive"]   = df_draft["position"].isin(DEFENSIVE_POSITIONS).astype(int)
    df_draft["is_qb"]          = (df_draft["position"] == "QB").astype(int)
    df_draft["is_first_round"] = (df_draft["round"] == 1).astype(int)

    # Aggregate to one row per team
    draft_summary = (
        df_draft.groupby("team_abbr").agg(
            total_draft_score = ("draft_score",    "sum"),
            num_picks         = ("round",          "count"),
            first_round_picks = ("is_first_round", "sum"),
            offensive_picks   = ("is_offensive",   "sum"),
            defensive_picks   = ("is_defensive",   "sum"),
            qb_drafted        = ("is_qb",          "max"),
        )
        .reset_index()
        .rename(columns={"team_abbr": "team"})
    )
    print(f"  Draft aggregated for {len(draft_summary)} teams")

    # Merge all three sources (left join keeps all 32 teams from stats)
    df = df_stats.merge(draft_summary, on="team", how="left")
    df = df.merge(df_trades,           on="team", how="left")

    draft_cols = ["total_draft_score","num_picks","first_round_picks",
                  "offensive_picks","defensive_picks","qb_drafted"]
    df[draft_cols]    = df[draft_cols].fillna(0).astype(int)
    df["trade_score"] = df["trade_score"].fillna(0)

    print(f"  Merged dataset: {df.shape[0]} teams x {df.shape[1]} columns\n")
    return df


# =============================================================================
# STEP 2 — engineer_features
# =============================================================================

def engineer_features(df):
    """
    Add derived features to the merged DataFrame.
    Returns (augmented_df, feature_name_list).
    """
    # -- Performance ratios --
    df["win_pct_last"]            = df["wins_last"] / (df["wins_last"] + df["losses_last"])
    df["points_per_game"]         = df["points_for"]     / 17
    df["points_allowed_per_game"] = df["points_against"] / 17
    df["net_points_per_game"]     = df["point_diff"]     / 17  # ppg - papg

    # -- Draft quality metrics --
    # Avoid division by zero for teams with 0 picks
    df["draft_efficiency"] = np.where(
        df["num_picks"] > 0, df["total_draft_score"] / df["num_picks"], 0
    )
    df["early_pick_ratio"] = np.where(
        df["num_picks"] > 0, df["first_round_picks"] / df["num_picks"], 0
    )

    # -- Combined roster movement index --
    df["roster_change_index"] = df["total_draft_score"] + df["trade_score"]

    # -- Binary team-status flags --
    df["elite_team_last_year"] = (df["wins_last"] >= 11).astype(int)  # 1 if likely a contender
    df["rebuild_team"]         = (df["wins_last"] <= 6).astype(int)   # 1 if in rebuild mode

    features = [
        # Core team performance
        "wins_last",              "win_pct_last",
        "net_points_per_game",    "points_per_game",        "points_allowed_per_game",
        "offensive_rank",         "defensive_rank",          "strength_of_schedule",
        # Team status flags
        "elite_team_last_year",   "rebuild_team",
        # Draft features
        "total_draft_score",      "num_picks",               "first_round_picks",
        "draft_efficiency",       "early_pick_ratio",
        "offensive_picks",        "defensive_picks",         "qb_drafted",
        # Roster movement
        "trade_score",            "roster_change_index",
    ]

    print(f"Feature engineering: {len(features)} total features")
    new = ["win_pct_last","net_points_per_game","points_per_game",
           "points_allowed_per_game","draft_efficiency","early_pick_ratio",
           "roster_change_index","elite_team_last_year","rebuild_team"]
    print(f"  New features: {', '.join(new)}\n")
    return df, features


# =============================================================================
# STEP 3 — train_models
# =============================================================================

def train_models(X_train, y_train):
    """
    Train all regression models on the training set.
    Linear Regression is wrapped in a StandardScaler pipeline so that
    features are normalized before fitting.
    Returns dict {model_name: fitted_model}.
    """
    models = {}

    # StandardScaler normalizes features to mean=0, std=1 — important for
    # linear regression so that large-scale features don't dominate small ones
    models["Linear Regression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression()),
    ]).fit(X_train, y_train)

    models["Random Forest"] = RandomForestRegressor(
        n_estimators=100, random_state=42
    ).fit(X_train, y_train)

    # Gradient Boosting builds trees sequentially, each correcting prior errors
    models["Gradient Boosting"] = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
    ).fit(X_train, y_train)

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            random_state=42, verbosity=0
        ).fit(X_train, y_train)

    print(f"Trained: {', '.join(models.keys())}\n")
    return models


# =============================================================================
# STEP 4 — evaluate_models
# =============================================================================

def evaluate_models(models, X_train, y_train, X_test, y_test, X_all, y_all):
    """
    Evaluate all models on the held-out test set and with 5-fold CV.
    Prints a clean comparison table.
    Model selection uses CV MAE — more reliable than a single split
    when n=32 (8 test teams is a very small evaluation set).
    Returns (results_df, best_model, best_name).
    """
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    rows  = []

    for name, model in models.items():
        train_mae  = mean_absolute_error(y_train, model.predict(X_train))
        test_preds = model.predict(X_test)
        test_mae   = mean_absolute_error(y_test, test_preds)
        test_rmse  = np.sqrt(mean_squared_error(y_test, test_preds))
        test_r2    = r2_score(y_test, test_preds)
        cv_scores  = cross_val_score(model, X_all, y_all, cv=kfold,
                                     scoring="neg_mean_absolute_error")
        cv_mae = -cv_scores.mean()
        rows.append(dict(
            Model=name,
            TrainMAE=round(train_mae, 2), TestMAE=round(test_mae, 2),
            TestRMSE=round(test_rmse, 2), TestR2=round(test_r2, 3),
            CV_MAE=round(cv_mae, 2),
        ))

    results_df = pd.DataFrame(rows).set_index("Model")

    # Print comparison table
    W = 72
    print("=" * W)
    print(f"{'MODEL COMPARISON  (25% held-out test  +  5-fold CV)':^{W}}")
    print("=" * W)
    hdr = f"{'Model':<24} {'Train':>8} {'Test':>8} {'Test':>8} {'Test':>8} {'5-fold':>8}"
    sub = f"{'':24} {'MAE':>8} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'CV MAE':>8}"
    print(hdr)
    print(sub)
    print("-" * W)
    for name, row in results_df.iterrows():
        print(f"{name:<24} {row['TrainMAE']:>8.2f} {row['TestMAE']:>8.2f} "
              f"{row['TestRMSE']:>8.2f} {row['TestR2']:>8.3f} {row['CV_MAE']:>8.2f}")
    print("=" * W)

    # Select best model by CV MAE (most robust metric with small n)
    best_name  = results_df["CV_MAE"].idxmin()
    best_model = models[best_name]
    print(f"\nBest model (lowest 5-fold CV MAE): {best_name}")

    # Feature importances for every model
    print()
    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=X_all.columns)
        elif hasattr(model, "named_steps"):
            coef = model.named_steps["model"].coef_
            imp  = pd.Series(np.abs(coef), index=X_all.columns)
        else:
            continue
        top5 = imp.sort_values(ascending=False).head(5)
        label = "Feature importances" if hasattr(model, "feature_importances_") else "|Coefficients|"
        print(f"  {label} [{name}]:")
        for feat, val in top5.items():
            print(f"    {feat:<32} {val:.4f}")
        print()

    return results_df, best_model, best_name


# =============================================================================
# STEP 5 — generate_predictions
# =============================================================================

def generate_predictions(df, best_model, features, best_name="Model"):
    """
    Retrain the best model on all 32 teams (full dataset), predict 2026 wins,
    clip to [0, 17], and return (df_with_predictions, df_results_sorted).
    """
    X_all = df[features]
    y_all = df[TARGET]

    best_model.fit(X_all, y_all)   # retrain on full data before final predictions
    df["predicted_wins_2026"] = best_model.predict(X_all).clip(0, 17).round(1)
    df["win_change"]          = (df["predicted_wins_2026"] - df["wins_last"]).round(1)

    df_results = (
        df[["team","predicted_wins_2026","win_change","wins_last",
            "total_draft_score","trade_score"]]
        .sort_values("predicted_wins_2026", ascending=False)
        .reset_index(drop=True)
    )
    df_results.insert(0, "rank", range(1, len(df_results) + 1))

    # Print ranked table
    W = 72
    print("\n" + "=" * W)
    print(f"{'NFL 2026 PREDICTED SEASON RANKINGS':^{W}}")
    print(f"{'(Best model: ' + best_name + ')':^{W}}")
    print("=" * W)
    print(f"{'Rank':<6}{'Team':<7}{'Pred W':>8}{'2025 W':>8}{'Change':>8}{'Draft':>8}{'Trade':>7}")
    print("-" * W)
    for _, row in df_results.iterrows():
        chg = f"+{row['win_change']:.1f}" if row["win_change"] >= 0 else f"{row['win_change']:.1f}"
        print(f"{int(row['rank']):<6}{row['team']:<7}"
              f"{row['predicted_wins_2026']:>8.1f}"
              f"{row['wins_last']:>8.0f}"
              f"{chg:>8}"
              f"{int(row['total_draft_score']):>8}"
              f"{int(row['trade_score']):>7}")
    print("=" * W)
    return df, df_results


# =============================================================================
# STEP 6 — print_insights
# =============================================================================

def print_insights(df, df_results):
    """
    Print key analytical takeaways:
      - Top 5 most improved teams
      - Top 5 most declining teams
      - Top 5 teams by draft score
      - Correlation analysis
    """
    W = 62
    print("\n" + "=" * W)
    print(f"{'KEY INSIGHTS':^{W}}")
    print("=" * W)

    # Most improved
    improved = df_results.nlargest(5, "win_change")
    print("\n  TOP 5 MOST IMPROVED TEAMS (predicted 2025 -> 2026):")
    print(f"  {'Team':<7}{'2025':>6}{'2026':>7}{'Change':>8}")
    print(f"  {'-'*29}")
    for _, r in improved.iterrows():
        chg = f"+{r['win_change']:.1f}" if r["win_change"] >= 0 else f"{r['win_change']:.1f}"
        print(f"  {r['team']:<7}{r['wins_last']:>6.0f}{r['predicted_wins_2026']:>7.1f}{chg:>8}")

    # Most declining
    declining = df_results.nsmallest(5, "win_change")
    print("\n  TOP 5 MOST DECLINING TEAMS (predicted 2025 -> 2026):")
    print(f"  {'Team':<7}{'2025':>6}{'2026':>7}{'Change':>8}")
    print(f"  {'-'*29}")
    for _, r in declining.iterrows():
        chg = f"+{r['win_change']:.1f}" if r["win_change"] >= 0 else f"{r['win_change']:.1f}"
        print(f"  {r['team']:<7}{r['wins_last']:>6.0f}{r['predicted_wins_2026']:>7.1f}{chg:>8}")

    # Draft leaders
    draft_top = df_results.nlargest(5, "total_draft_score")
    print("\n  TOP 5 TEAMS BY DRAFT SCORE:")
    print(f"  {'Team':<7}{'Draft Score':>12}{'Pred Wins':>11}")
    print(f"  {'-'*31}")
    for _, r in draft_top.iterrows():
        print(f"  {r['team']:<7}{int(r['total_draft_score']):>12}{r['predicted_wins_2026']:>11.1f}")

    # Correlation analysis
    r_wins   = df["wins_last"].corr(df["predicted_wins_2026"])
    r_draft  = df["total_draft_score"].corr(df["predicted_wins_2026"])
    r_trade  = df["trade_score"].corr(df["predicted_wins_2026"])
    r_eff    = df["draft_efficiency"].corr(df["predicted_wins_2026"])
    print("\n  PEARSON CORRELATIONS WITH PREDICTED 2026 WINS:")
    print(f"  {'wins_last':30} r = {r_wins:+.3f}  (strongest signal)")
    print(f"  {'total_draft_score':30} r = {r_draft:+.3f}")
    print(f"  {'draft_efficiency':30} r = {r_eff:+.3f}")
    print(f"  {'trade_score':30} r = {r_trade:+.3f}")
    print(f"\n  Note: Prior season wins (r={r_wins:.2f}) dominates short-term forecasts.")
    print(f"  Draft score (r={r_draft:.2f}) provides additional signal, especially")
    print(f"  for rebuilding teams making big draft-capital investments.")

    print("=" * W + "\n")


# =============================================================================
# STEP 7 — visualize_results
# =============================================================================

def visualize_results(df, df_results, best_model, best_name, features):
    """
    Save four charts to outputs/:
      1. predicted_wins_chart.png   - horizontal bar chart (all 32 teams)
      2. feature_importance.png     - ranked feature importances
      3. draft_vs_wins.png          - scatter: draft score vs predicted wins
      4. wins_comparison.png        - scatter: last year wins vs predicted wins
    """

    # ── Chart 1: Predicted wins bar chart ─────────────────────────────────────
    teams_rev = df_results["team"].tolist()[::-1]
    wins_rev  = df_results["predicted_wins_2026"].tolist()[::-1]
    colors    = plt.cm.RdYlGn(np.linspace(0.15, 0.9, len(teams_rev)))

    fig, ax = plt.subplots(figsize=(14, 11))
    bars = ax.barh(teams_rev, wins_rev, color=colors, edgecolor="white", height=0.75)
    ax.set_xlabel("Predicted Wins (2026)", fontsize=12)
    ax.set_title("NFL 2026 Predicted Wins by Team", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlim(0, 18)
    ax.axvline(8.5, color="gray", linestyle="--", lw=1.2, alpha=0.6, label=".500 pace")
    ax.legend(loc="lower right", fontsize=10)
    for bar, val in zip(bars, wins_rev):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=9)
    plt.tight_layout()
    _save_fig(fig, "predicted_wins_chart.png")

    # ── Chart 2: Feature importance ────────────────────────────────────────────
    if hasattr(best_model, "feature_importances_"):
        imp_vals = best_model.feature_importances_
        imp_label = f"Feature Importance — {best_name}"
        x_label   = "Importance Score"
    elif hasattr(best_model, "named_steps"):
        imp_vals  = np.abs(best_model.named_steps["model"].coef_)
        imp_label = f"Feature Importance (|Coefficient|) — {best_name}"
        x_label   = "|Scaled Coefficient|"
    else:
        imp_vals = None

    if imp_vals is not None:
        imp_series = pd.Series(imp_vals, index=features).sort_values(ascending=True)
        n_feats    = len(imp_series)
        bar_colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_feats))

        fig, ax = plt.subplots(figsize=(10, max(7, n_feats * 0.35)))
        ax.barh(imp_series.index, imp_series.values, color=bar_colors, edgecolor="white")
        ax.set_title(imp_label, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(x_label, fontsize=11)

        # Annotate the top 3 bars
        for i, (feat, val) in enumerate(imp_series.items()):
            if i >= n_feats - 3:
                ax.text(val + 0.001, i, f"{val:.3f}", va="center", fontsize=8)

        plt.tight_layout()
        _save_fig(fig, "feature_importance.png")

    # ── Chart 3: Draft score vs Predicted wins scatter ─────────────────────────
    r_draft = df["total_draft_score"].corr(df["predicted_wins_2026"])

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        df_results["total_draft_score"], df_results["predicted_wins_2026"],
        c=df_results["predicted_wins_2026"], cmap="RdYlGn",
        s=110, edgecolors="gray", linewidth=0.5, zorder=3,
    )
    for _, row in df_results.iterrows():
        ax.annotate(
            row["team"],
            (row["total_draft_score"], row["predicted_wins_2026"]),
            textcoords="offset points", xytext=(5, 4), fontsize=8, color="black",
        )
    # Linear trend line
    x_vals = df_results["total_draft_score"].values
    z      = np.polyfit(x_vals, df_results["predicted_wins_2026"].values, 1)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "b--", lw=1.5, alpha=0.6, label="Trend line")
    ax.set_xlabel("Total Draft Score", fontsize=12)
    ax.set_ylabel("Predicted 2026 Wins", fontsize=12)
    ax.set_title(f"Draft Score vs Predicted 2026 Wins  (r = {r_draft:.2f})",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    plt.colorbar(sc, ax=ax, label="Predicted Wins")
    plt.tight_layout()
    _save_fig(fig, "draft_vs_wins.png")

    # ── Chart 4: Last year wins vs Predicted wins scatter ──────────────────────
    r_wins = df["wins_last"].corr(df["predicted_wins_2026"])

    fig, ax = plt.subplots(figsize=(10, 7))
    sc2 = ax.scatter(
        df_results["wins_last"], df_results["predicted_wins_2026"],
        c=df_results["win_change"], cmap="RdYlGn",
        s=110, edgecolors="gray", linewidth=0.5, zorder=3, vmin=-4, vmax=4,
    )
    for _, row in df_results.iterrows():
        ax.annotate(
            row["team"],
            (row["wins_last"], row["predicted_wins_2026"]),
            textcoords="offset points", xytext=(5, 4), fontsize=8, color="black",
        )
    # y = x diagonal (no change reference line)
    ax.plot([0, 17], [0, 17], color="gray", linestyle="--", lw=1.2, alpha=0.5, label="No change")
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 17)
    ax.set_xlabel("2025 Wins (Last Season)", fontsize=12)
    ax.set_ylabel("Predicted 2026 Wins", fontsize=12)
    ax.set_title(
        f"2025 Wins vs Predicted 2026 Wins  (r = {r_wins:.2f})\nColor = projected win change",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=10)
    plt.colorbar(sc2, ax=ax, label="Predicted Win Change")
    plt.tight_layout()
    _save_fig(fig, "wins_comparison.png")

    print("Charts saved:")
    print("  outputs/predicted_wins_chart.png")
    print("  outputs/feature_importance.png")
    print("  outputs/draft_vs_wins.png")
    print("  outputs/wins_comparison.png")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

# 1. Load & merge all data sources
df = load_data()

# 2. Add engineered features; get the feature list back
df, FEATURES = engineer_features(df)

# 3. Prepare X and y matrices
X = df[FEATURES]
y = df[TARGET]

# 75 / 25 train-test split (fixed seed for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4. Train all models on the training set
models = train_models(X_train, y_train)

# 5. Compare models; pick the best by 5-fold CV MAE
eval_df, best_model, best_name = evaluate_models(
    models, X_train, y_train, X_test, y_test, X, y
)

# 6. Retrain best model on full data; generate 2026 predictions
df, df_results = generate_predictions(df, best_model, FEATURES, best_name)

# 7. Save predictions CSV
pred_file = OUTPUT_DIR / "predictions_2026.csv"
df_results.to_csv(pred_file, index=False)
print(f"\nSaved: {pred_file}")

# 8. Print key insights
print_insights(df, df_results)

# 9. Save all charts
visualize_results(df, df_results, best_model, best_name, FEATURES)

print("Done! All outputs are in the outputs/ folder.")
