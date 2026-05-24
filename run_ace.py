"""
run_ace.py — Project ACE, single-file unified engine.

Workflow:
  1. Load all real Sackmann ATP CSVs from the data/ folder.
  2. Parse actual set scores to build the training target.
  3. Compute per-player rolling Elo, SDI, and Fatigue from real stats.
  4. Train XGBoost on those features with time-series ordering.
  5. Look up the two named players, extract their current feature values.
  6. Predict set-score probabilities and search for +EV Dutching opportunities.
"""

import os
import glob
import math
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from dutcher import derive_market_probs, find_optimal_dutch
from feature_builder import (
    compute_double_fault_rate,
    compute_days_since_last_match,
    compute_tournament_tier,
    compute_ranking_difference,
)

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── 1. LOAD CSVs ─────────────────────────────────────────────────────────────
def load_atp_data():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "atp_matches_*.csv")))
    if not files:
        raise FileNotFoundError(
            "No ATP match CSVs found in data/. Run 'python ingest_data.py' first."
        )
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception:
            pass
    df = pd.concat(dfs, ignore_index=True)
    return df


def load_ranking_data():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "atp_rankings_*.csv")))
    if not files:
        return pd.DataFrame(columns=["ranking_date", "player", "rank", "points"])

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(
                f,
                low_memory=False,
                usecols=["ranking_date", "player", "rank", "points"],
            ))
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame(columns=["ranking_date", "player", "rank", "points"])

    ranking_df = pd.concat(dfs, ignore_index=True)
    ranking_df["ranking_date"] = pd.to_numeric(ranking_df["ranking_date"], errors="coerce").astype("Int64")
    ranking_df["player"] = pd.to_numeric(ranking_df["player"], errors="coerce").astype("Int64")
    ranking_df["rank"] = pd.to_numeric(ranking_df["rank"], errors="coerce").astype("Int64")
    ranking_df["points"] = pd.to_numeric(ranking_df["points"], errors="coerce")
    ranking_df = ranking_df.dropna(subset=["ranking_date", "player"])
    ranking_df = ranking_df.sort_values(["player", "ranking_date"], kind="stable").reset_index(drop=True)
    return ranking_df


def enrich_match_ranks(df, ranking_df):
    if ranking_df.empty:
        return df

    ranking_df = ranking_df.rename(columns={"player": "player_id"})
    ranking_df = ranking_df[["player_id", "ranking_date", "rank", "points"]]

    df = df.copy()
    df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce")
    df["winner_id"] = pd.to_numeric(df["winner_id"], errors="coerce").astype("Int64")
    df["loser_id"] = pd.to_numeric(df["loser_id"], errors="coerce").astype("Int64")

    base = df.reset_index().rename(columns={"index": "orig_index"})
    ranked = ranking_df.sort_values(["ranking_date", "player_id"]).copy()
    ranked = ranked.dropna(subset=["player_id", "ranking_date"])
    ranked["player_id"] = ranked["player_id"].astype("int64")
    ranked["ranking_date"] = ranked["ranking_date"].astype("int64")

    winner_left = base[["orig_index", "tourney_date", "winner_id"]].rename(columns={"winner_id": "player_id"}).copy()
    winner_left = winner_left.dropna(subset=["player_id", "tourney_date"]) \
        .astype({"player_id": "int64", "tourney_date": "int64"})
    winner_left = winner_left.sort_values(["tourney_date", "player_id"])
    winner_lookup = pd.merge_asof(
        winner_left,
        ranked,
        left_on="tourney_date",
        right_on="ranking_date",
        by="player_id",
        direction="backward",
    ).set_index("orig_index")

    loser_left = base[["orig_index", "tourney_date", "loser_id"]].rename(columns={"loser_id": "player_id"}).copy()
    loser_left = loser_left.dropna(subset=["player_id", "tourney_date"]) \
        .astype({"player_id": "int64", "tourney_date": "int64"})
    loser_left = loser_left.sort_values(["tourney_date", "player_id"])
    loser_lookup = pd.merge_asof(
        loser_left,
        ranked,
        left_on="tourney_date",
        right_on="ranking_date",
        by="player_id",
        direction="backward",
    ).set_index("orig_index")

    base["winner_rank_lookup"] = base["orig_index"].map(winner_lookup["rank"]).fillna(200)
    base["loser_rank_lookup"] = base["orig_index"].map(loser_lookup["rank"]).fillna(200)
    base["winner_rank_points_lookup"] = base["orig_index"].map(winner_lookup["points"]).fillna(0)
    base["loser_rank_points_lookup"] = base["orig_index"].map(loser_lookup["points"]).fillna(0)

    base["winner_rank"] = base["winner_rank"].fillna(base["winner_rank_lookup"]).fillna(200).astype(int)
    base["loser_rank"] = base["loser_rank"].fillna(base["loser_rank_lookup"]).fillna(200).astype(int)
    base["winner_rank_points"] = base.get("winner_rank_points", base["winner_rank_points_lookup"]).fillna(base["winner_rank_points_lookup"]).fillna(0)
    base["loser_rank_points"] = base.get("loser_rank_points", base["loser_rank_points_lookup"]).fillna(base["loser_rank_points_lookup"]).fillna(0)

    return base.sort_values("orig_index").drop(columns=["orig_index"])

# ── 2. PARSE SET SCORES → TARGET ─────────────────────────────────────────────
def parse_target(score, best_of):
    """Return 0=2-0, 1=2-1, 2=0-2, 3=1-2, or -1 if unparseable."""
    if not isinstance(score, str) or pd.isna(score):
        return -1
    # Remove retirements, walkovers, etc.
    for tag in ["RET", "W/O", "DEF", "ABN", "UNK"]:
        if tag in score.upper():
            return -1
    try:
        sets = score.strip().split()
        w_sets = l_sets = 0
        for s in sets:
            # handle super-tiebreak notation like "10-5" treated as a set
            parts = s.replace("(", "").split("(")[0].split("-")
            if len(parts) != 2:
                continue
            wg, lg = int(parts[0]), int(parts[1])
            if wg > lg:
                w_sets += 1
            else:
                l_sets += 1
        if w_sets == 2 and l_sets == 0:
            return 0  # 2-0
        elif w_sets == 2 and l_sets == 1:
            return 1  # 2-1
        elif w_sets == 0 and l_sets == 2:
            return 2  # 0-2
        elif w_sets == 1 and l_sets == 2:
            return 3  # 1-2
        return -1
    except Exception:
        return -1

# ── 3. ROLLING ELO ───────────────────────────────────────────────────────────
def compute_elo_ratings(df):
    """Add winner_elo and loser_elo columns using a simple 1500-base Elo."""
    ratings = {}

    def get_rating(name):
        return ratings.get(name, 1500.0)

    def k_factor(n_matches):
        return 32.0 if n_matches < 30 else 20.0

    match_counts = {}
    w_elos, l_elos = [], []

    for _, row in df.iterrows():
        w, l = row["winner_name"], row["loser_name"]
        if not isinstance(w, str) or not isinstance(l, str):
            w_elos.append(np.nan)
            l_elos.append(np.nan)
            continue

        r_w, r_l = get_rating(w), get_rating(l)
        w_elos.append(r_w)
        l_elos.append(r_l)

        exp_w = 1.0 / (1.0 + 10 ** ((r_l - r_w) / 400.0))
        k_w = k_factor(match_counts.get(w, 0))
        k_l = k_factor(match_counts.get(l, 0))

        ratings[w] = r_w + k_w * (1.0 - exp_w)
        ratings[l] = r_l + k_l * (0.0 - (1.0 - exp_w))

        match_counts[w] = match_counts.get(w, 0) + 1
        match_counts[l] = match_counts.get(l, 0) + 1

    df["winner_elo"] = w_elos
    df["loser_elo"] = l_elos
    return df, ratings  # return final ratings for live lookup

# ── 4. FEATURE EXTRACTION ────────────────────────────────────────────────────
def build_features(df):
    """Build model features from match rows. Returns (X, y)."""
    rows = []
    targets = []

    last_match_date = {}
    h2h_totals = {}
    h2h_wins = {}

    df = df.sort_values("tourney_date").reset_index(drop=True)
    for _, row in df.iterrows():
        target = row.get("target", -1)
        if target == -1:
            continue

        winner = row.get("winner_name") or ""
        loser = row.get("loser_name") or ""
        w_svpt = row.get("w_svpt", 0) or 0
        l_svpt = row.get("l_svpt", 0) or 0
        w_1stIn = row.get("w_1stIn", 0) or 0
        l_1stIn = row.get("l_1stIn", 0) or 0
        w_1stWon = row.get("w_1stWon", 0) or 0
        l_1stWon = row.get("l_1stWon", 0) or 0
        w_ace = row.get("w_ace", 0) or 0
        l_ace = row.get("l_ace", 0) or 0
        w_df = row.get("w_df", 0) or 0
        l_df = row.get("l_df", 0) or 0
        w_rank = int(row.get("winner_rank", 200) or 200)
        l_rank = int(row.get("loser_rank", 200) or 200)
        w_rank_pts = float(row.get("winner_rank_points", 0) or 0)
        l_rank_pts = float(row.get("loser_rank_points", 0) or 0)
        tourney_tier = compute_tournament_tier(row.get("tourney_level"))

        w_sdi = (w_ace / w_svpt if w_svpt > 0 else 0) + (w_1stWon / w_1stIn if w_1stIn > 0 else 0)
        l_sdi = (l_ace / l_svpt if l_svpt > 0 else 0) + (l_1stWon / l_1stIn if l_1stIn > 0 else 0)

        current_date = pd.to_datetime(row.get("tourney_date", np.nan), format="%Y%m%d", errors="coerce")
        w_last_date = last_match_date.get(winner)
        l_last_date = last_match_date.get(loser)
        w_days_since = (current_date - w_last_date).days if pd.notna(current_date) and w_last_date is not None else np.nan
        l_days_since = (current_date - l_last_date).days if pd.notna(current_date) and l_last_date is not None else np.nan

        names_pair = tuple(sorted([winner, loser]))
        total_pair = h2h_totals.get(names_pair, 0)
        winner_pair_wins = h2h_wins.get((winner, loser), 0)
        loser_pair_wins = h2h_wins.get((loser, winner), 0)
        w_h2h = winner_pair_wins / total_pair if total_pair > 0 else 0.5
        l_h2h = loser_pair_wins / total_pair if total_pair > 0 else 0.5

        winner_elo = row.get("winner_elo", 1500)
        loser_elo = row.get("loser_elo", 1500)
        if pd.isna(winner_elo):
            winner_elo = 1500
        if pd.isna(loser_elo):
            loser_elo = 1500

        rows.append({
            "winner_elo": winner_elo,
            "loser_elo": loser_elo,
            "elo_diff": winner_elo - loser_elo,
            "winner_sdi": w_sdi,
            "loser_sdi": l_sdi,
            "sdi_diff": w_sdi - l_sdi,
            "winner_rank": w_rank,
            "loser_rank": l_rank,
            "rank_diff": compute_ranking_difference(w_rank, l_rank),
            "winner_rank_points": w_rank_pts,
            "loser_rank_points": l_rank_pts,
            "rank_points_diff": w_rank_pts - l_rank_pts,
            "tourney_tier": tourney_tier,
            "winner_double_fault_rate": compute_double_fault_rate(w_df, w_svpt),
            "loser_double_fault_rate": compute_double_fault_rate(l_df, l_svpt),
            "double_fault_diff": compute_double_fault_rate(l_df, l_svpt) - compute_double_fault_rate(w_df, w_svpt),
            "winner_days_since_last": w_days_since if not np.isnan(w_days_since) else 30,
            "loser_days_since_last": l_days_since if not np.isnan(l_days_since) else 30,
            "days_since_last_diff": (w_days_since if not np.isnan(w_days_since) else 30) - (l_days_since if not np.isnan(l_days_since) else 30),
            "winner_h2h": w_h2h,
            "loser_h2h": l_h2h,
            "h2h_diff": w_h2h - l_h2h,
            "year": int(current_date.year) if pd.notna(current_date) else 0,
        })
        targets.append(int(target))

        # Update historic trackers after recording the row
        if pd.notna(current_date):
            last_match_date[winner] = current_date
            last_match_date[loser] = current_date
        h2h_totals[names_pair] = total_pair + 1
        h2h_wins[(winner, loser)] = winner_pair_wins + 1

    X = pd.DataFrame(rows).fillna(0)
    y = np.array(targets)
    return X, y

# ── 5. LIVE FEATURES FOR A SPECIFIC PLAYER ───────────────────────────────────
def get_player_live_features(df, player_name, final_ratings):
    """Get the most recent live feature values for a player."""
    # defensively handle missing/non-string player_name
    pname = player_name if isinstance(player_name, str) else ""
    pname_l = pname.lower().strip()
    w_rows = df[df["winner_name"].fillna("").str.lower() == pname_l]
    l_rows = df[df["loser_name"].fillna("").str.lower() == pname_l]

    elo = final_ratings.get(pname, 1500.0)
    rank = 200
    rank_points = 0.0
    double_fault_rate = 0.05
    days_since_last = 30
    sdi = 0.5

    if not w_rows.empty:
        last = w_rows.iloc[-1]
        svpt = last.get("w_svpt", 0) or 0
        first_in = last.get("w_1stIn", 0) or 0
        first_won = last.get("w_1stWon", 0) or 0
        ace = last.get("w_ace", 0) or 0
        df_val = last.get("w_df", 0) or 0
        # read values without forcing pandas to coerce types
        rnk = last.get("winner_rank")
        if pd.isna(rnk):
            rank = 200
        else:
            try:
                rank = int(rnk)
            except Exception:
                rank = 200
        rnkpts = last.get("winner_rank_points")
        if pd.isna(rnkpts):
            rank_points = 0.0
        else:
            try:
                rank_points = float(rnkpts)
            except Exception:
                rank_points = 0.0
        sdi = (ace / svpt if svpt > 0 else 0) + (first_won / first_in if first_in > 0 else 0)
        double_fault_rate = compute_double_fault_rate(df_val, svpt)
        last_date = pd.to_datetime(last.get("tourney_date", np.nan), format="%Y%m%d", errors="coerce")
        days_since_last = 0 if pd.isna(last_date) else 0
    elif not l_rows.empty:
        last = l_rows.iloc[-1]
        svpt = last.get("l_svpt", 0) or 0
        first_in = last.get("l_1stIn", 0) or 0
        first_won = last.get("l_1stWon", 0) or 0
        ace = last.get("l_ace", 0) or 0
        df_val = last.get("l_df", 0) or 0
        rnk = last.get("loser_rank")
        if pd.isna(rnk):
            rank = 200
        else:
            try:
                rank = int(rnk)
            except Exception:
                rank = 200
        rnkpts = last.get("loser_rank_points")
        if pd.isna(rnkpts):
            rank_points = 0.0
        else:
            try:
                rank_points = float(rnkpts)
            except Exception:
                rank_points = 0.0
        sdi = (ace / svpt if svpt > 0 else 0) + (first_won / first_in if first_in > 0 else 0)
        double_fault_rate = compute_double_fault_rate(df_val, svpt)
        last_date = pd.to_datetime(last.get("tourney_date", np.nan), format="%Y%m%d", errors="coerce")
        days_since_last = 0 if pd.isna(last_date) else 0

    return {
        "elo": elo,
        "sdi": sdi,
        "rank": rank,
        "rank_points": rank_points,
        "double_fault_rate": double_fault_rate,
        "days_since_last": days_since_last,
    }


def compute_live_h2h(df, p1, p2, date_int=None):
    """Compute live head-to-head (p1 vs p2) using historical matches before date_int.

    Returns (p1_h2h, p2_h2h) where p1_h2h = wins_p1 / total_matches.
    If no historical meetings, returns (0.5, 0.5).
    """
    if date_int is None:
        # use latest available date in df
        try:
            date_int = int(df["tourney_date"].max())
        except Exception:
            date_int = None

    hist = df.copy()
    hist["tourney_date"] = pd.to_numeric(hist["tourney_date"], errors="coerce")
    if date_int is not None:
        hist = hist[hist["tourney_date"] < date_int]

    # handle non-string names
    if not isinstance(p1, str) or not isinstance(p2, str):
        return 0.5, 0.5
    p1_l = p1.lower().strip()
    p2_l = p2.lower().strip()

    p1_vs_p2 = ((hist["winner_name"].str.lower() == p1_l) & (hist["loser_name"].str.lower() == p2_l))
    p2_vs_p1 = ((hist["winner_name"].str.lower() == p2_l) & (hist["loser_name"].str.lower() == p1_l))

    w1 = int(p1_vs_p2.sum())
    w2 = int(p2_vs_p1.sum())
    total = w1 + w2
    if total == 0:
        return 0.5, 0.5
    return float(w1 / total), float(w2 / total)


def load_training_data():
    df = load_atp_data()
    ranking_df = load_ranking_data()
    df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce")
    df = df.sort_values("tourney_date").reset_index(drop=True)
    df = enrich_match_ranks(df, ranking_df)
    df = df.dropna(subset=["tourney_date"]).copy()
    df["year"] = (df["tourney_date"] // 10000).astype(int)
    df["target"] = df.apply(lambda r: parse_target(r.get("score", ""), r.get("best_of", 3)), axis=1)
    df, _ = compute_elo_ratings(df)
    X, y = build_features(df[df["target"] >= 0])
    return df, X, y


def evaluate_time_series_cv(X, y):
    """Evaluate time-series cross-validation by year."""
    if "year" not in X.columns:
        return np.nan
    years = sorted(X["year"].unique())
    if len(years) < 2:
        return np.nan

    cv_losses = []
    for i in range(1, len(years)):
        train_mask = X["year"] < years[i]
        val_mask = X["year"] == years[i]
        if train_mask.sum() < 50 or val_mask.sum() == 0:
            continue

        X_train = X[train_mask].drop(columns=["year"])
        y_train = y[train_mask]
        X_val = X[val_mask].drop(columns=["year"])
        y_val = y[val_mask]

        model = xgb.XGBClassifier(
            max_depth=4, learning_rate=0.05, n_estimators=300,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob",
            verbosity=0, use_label_encoder=False
        )
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)
        # Align predicted probability columns to full 4-class ordering using
        # model.classes_ (which maps column order -> original labels).
        try:
            classes = getattr(model, "classes_", None)
            n_classes = 4
            if classes is None:
                # If classes_ not available, assume preds already match 4 cols
                if preds.shape[1] != n_classes:
                    raise ValueError("Unexpected pred shape and missing classes_")
                full_preds = preds
            else:
                full_preds = np.zeros((preds.shape[0], n_classes), dtype=float)
                for col_idx, lbl in enumerate(classes):
                    lbl = int(lbl)
                    if 0 <= lbl < n_classes:
                        full_preds[:, lbl] = preds[:, col_idx]
                # If some columns remained zero (missing class), smooth and renormalize
                eps = 1e-15
                full_preds = np.clip(full_preds, eps, 1 - eps)
                full_preds = full_preds / full_preds.sum(axis=1, keepdims=True)

            ll = _multiclass_log_loss(y_val, full_preds, labels=[0, 1, 2, 3])
            cv_losses.append(ll)
        except Exception as e:
            print(f"[CV] Skipping fold {years[i]} due to error: {e}")
            continue

    return np.mean(cv_losses) if cv_losses else np.nan

# ── 6. NORMALIZE KEY (player name → Player 1 / Player 2) ─────────────────────
def normalize_key(key, p1, p2):
    k = key.strip()
    for name in [p1, p1.split()[-1]]:
        k = k.replace(name, "Player 1")
    for name in [p2, p2.split()[-1]]:
        k = k.replace(name, "Player 2")
    return k


def _multiclass_log_loss(y_true, y_pred, labels=None, eps=1e-15):
    """Compute multiclass log loss without sklearn validation overhead.

    y_true: array-like (n_samples,)
    y_pred: array-like (n_samples, n_classes) probabilities in column order
    labels: list of class labels corresponding to columns in y_pred
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=float)
    if labels is None:
        labels = np.arange(y_pred.shape[1])
    # Ensure shapes
    if y_pred.ndim != 2:
        raise ValueError("y_pred must be 2D (n_samples, n_classes)")
    if y_pred.shape[1] != len(labels):
        raise ValueError("Number of columns in y_pred must match number of labels")

    # Build one-hot true matrix aligned to labels
    one_hot = np.zeros_like(y_pred)
    for i, lbl in enumerate(labels):
        one_hot[:, i] = (y_true == lbl).astype(float)

    # Clip probabilities and renormalize
    y_pred = np.clip(y_pred, eps, 1 - eps)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)

    # Compute log-loss
    loss = - (one_hot * np.log(y_pred)).sum() / y_pred.shape[0]
    return float(loss)

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=========================================")
    print("      Project ACE - Unified Engine       ")
    print("=========================================")

    p1 = input("Enter Player 1 (e.g., Andrey Rublev): ").strip()
    p2 = input("Enter Player 2 (e.g., Hamad Medjedovic): ").strip()
    surface = input("Enter Surface (Hard/Clay/Grass): ").strip()

    print(f"\n[1/5] Loading Sackmann ATP data...")
    df = load_atp_data()
    ranking_df = load_ranking_data()

    df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce")
    df = df.sort_values("tourney_date").reset_index(drop=True)
    df = enrich_match_ranks(df, ranking_df)
    df["year"] = (df["tourney_date"] // 10000).astype(int)

    df["target"] = df.apply(lambda r: parse_target(r.get("score", ""), r.get("best_of", 3)), axis=1)
    print(f"    {len(df)} matches loaded, {(df['target']>=0).sum()} with valid set scores.")

    print(f"\n[2/5] Computing rolling Elo ratings...")
    df, final_ratings = compute_elo_ratings(df)

    print(f"\n[3/5] Building features and evaluating time-series CV...")
    X, y = build_features(df[df["target"] >= 0])
    cv_loss = evaluate_time_series_cv(X, y)
    if pd.notna(cv_loss):
        print(f"    Time-series CV log loss: {cv_loss:.4f}")
    else:
        print("    Not enough distinct years for cross-validation; training on all data.")

    features = X.drop(columns=["year"])
    model = xgb.XGBClassifier(
        max_depth=4, learning_rate=0.05, n_estimators=300,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=4,
        verbosity=0, use_label_encoder=False
    )
    model.fit(features, y)
    print(f"    Model trained on {len(features)} matches.")

    print(f"\n[4/5] Extracting live features for {p1} vs {p2}...")
    p1_features = get_player_live_features(df, p1, final_ratings)
    p2_features = get_player_live_features(df, p2, final_ratings)

    print(f"    {p1}: Elo={p1_features['elo']:.0f}, SDI={p1_features['sdi']:.3f}, Rank={p1_features['rank']}")
    print(f"    {p2}: Elo={p2_features['elo']:.0f}, SDI={p2_features['sdi']:.3f}, Rank={p2_features['rank']}")

    default_tier = compute_tournament_tier(None)
    # compute live head-to-head from historical matches
    recent_date = None
    try:
        recent_date = int(df["tourney_date"].max())
    except Exception:
        recent_date = None
    p1_h2h, p2_h2h = compute_live_h2h(df, p1, p2, recent_date)

    live_X = pd.DataFrame([{
        "winner_elo": p1_features["elo"],
        "loser_elo": p2_features["elo"],
        "elo_diff": p1_features["elo"] - p2_features["elo"],
        "winner_sdi": p1_features["sdi"],
        "loser_sdi": p2_features["sdi"],
        "sdi_diff": p1_features["sdi"] - p2_features["sdi"],
        "winner_rank": p1_features["rank"],
        "loser_rank": p2_features["rank"],
        "rank_diff": compute_ranking_difference(p1_features["rank"], p2_features["rank"]),
        "winner_rank_points": p1_features["rank_points"],
        "loser_rank_points": p2_features["rank_points"],
        "rank_points_diff": p1_features["rank_points"] - p2_features["rank_points"],
        "tourney_tier": default_tier,
        "winner_double_fault_rate": p1_features["double_fault_rate"],
        "loser_double_fault_rate": p2_features["double_fault_rate"],
        "double_fault_diff": p2_features["double_fault_rate"] - p1_features["double_fault_rate"],
        "winner_days_since_last": p1_features["days_since_last"],
        "loser_days_since_last": p2_features["days_since_last"],
        "days_since_last_diff": p1_features["days_since_last"] - p2_features["days_since_last"],
        "winner_h2h": p1_h2h,
        "loser_h2h": p2_h2h,
        "h2h_diff": p1_h2h - p2_h2h,
    }])

    raw_probs = model.predict_proba(live_X)[0]
    core_p = {"2-0": raw_probs[0], "2-1": raw_probs[1], "0-2": raw_probs[2], "1-2": raw_probs[3]}

    print(f"\n    AI Set-Score Probabilities:")
    print(f"      2-0 ({p1} sweeps): {core_p['2-0']*100:.1f}%")
    print(f"      2-1 ({p1} in 3):   {core_p['2-1']*100:.1f}%")
    print(f"      0-2 ({p2} sweeps): {core_p['0-2']*100:.1f}%")
    print(f"      1-2 ({p2} in 3):   {core_p['1-2']*100:.1f}%")

    derived_probs = derive_market_probs(core_p)
    supported_keys = set(derived_probs.keys())

    print(f"\nSupported markets:")
    for k in sorted(supported_keys):
        print(f"  * {k.replace('Player 1', p1).replace('Player 2', p2)}")

    # ── Betting loop ──────────────────────────────────────────────────────────
    while True:
        try:
            print("\n-----------------------------------------")
            print(f"FORMAT  --> Market=Odds, Market=Odds")
            print(f"EXAMPLE --> Match winner - {p1}=1.60, Total sets - over 2.5=2.22")
            print("-----------------------------------------")

            odds_inp = input("\nEnter Sportsbook Odds ('q' to quit) > ")
            if odds_inp.lower() == "q":
                break

            raw_odds = {}
            for pair in odds_inp.split(","):
                try:
                    k, v = pair.rsplit("=", 1)
                    raw_odds[k.strip()] = float(v.strip())
                except ValueError:
                    print(f"  [skip] Cannot parse: {pair.strip()}")

            m_odds = {}
            skipped = []
            for k, v in raw_odds.items():
                nk = normalize_key(k, p1, p2)
                if nk in supported_keys:
                    m_odds[nk] = v
                else:
                    skipped.append(k)

            if skipped:
                print(f"  [!] Skipped (not supported): {', '.join(skipped)}")
            if not m_odds:
                print("  [!] No recognized markets. Check spelling against the list above.")
                continue

            print(f"  [OK] Scanning {len(m_odds)} market(s)...")
            res = find_optimal_dutch(derived_probs, m_odds, bankroll=100, kelly_fraction=0.25)

            print("\n==== OPTIMAL DUTCHING RESULT ====")
            if res["decision"] == "BET":
                legs = [l.replace("Player 1", p1).replace("Player 2", p2) for l in res["covered_legs"]]
                print(f"[BET] EDGE FOUND! (+{res['edge']:.4f} vs the book)")
                print(f"Optimal Legs: {legs}")
                print(f"Stakes (from $100 bankroll):")
                for k, v in res["stakes"].items():
                    print(f"  --> {k.replace('Player 1', p1).replace('Player 2', p2)}: ${v:.2f}")
                print(f"Guaranteed return: ${res['guaranteed_return']:.2f}")
            else:
                print("[NO BET] No combination yields +EV at these odds.")
            print("=================================\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
