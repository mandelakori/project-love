import argparse
import pandas as pd
import numpy as np
import os

from dutcher import derive_market_probs, find_optimal_dutch
from run_ace import (
    load_atp_data,
    load_ranking_data,
    enrich_match_ranks,
    compute_elo_ratings,
    build_features,
    get_player_live_features,
    parse_target,
    compute_tournament_tier,
    compute_ranking_difference,
)
import xgboost as xgb


def load_and_prepare_data():
    print("Scanning data/ for real Sackmann ATP data...")
    df = load_atp_data()
    ranking_df = load_ranking_data()

    df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce")
    df = df.sort_values("tourney_date").reset_index(drop=True)
    df = enrich_match_ranks(df, ranking_df)
    df = df.dropna(subset=["tourney_date"]).copy()
    df["year"] = (df["tourney_date"] // 10000).astype(int)
    df["target"] = df.apply(lambda r: parse_target(r.get("score", ""), r.get("best_of", 3)), axis=1)

    df, final_ratings = compute_elo_ratings(df)
    X, y = build_features(df[df["target"] >= 0])
    return df, X, y, final_ratings


def extract_live_features(player_a, player_b, surface, df, final_ratings):
    print(f"Extracting live features for {player_a} vs {player_b} on {surface}...")
    p1 = get_player_live_features(df, player_a, final_ratings)
    p2 = get_player_live_features(df, player_b, final_ratings)

    default_tier = compute_tournament_tier(None)
    live_X = pd.DataFrame([{
        "winner_elo": p1["elo"],
        "loser_elo": p2["elo"],
        "elo_diff": p1["elo"] - p2["elo"],
        "winner_sdi": p1["sdi"],
        "loser_sdi": p2["sdi"],
        "sdi_diff": p1["sdi"] - p2["sdi"],
        "winner_rank": p1["rank"],
        "loser_rank": p2["rank"],
        "rank_diff": compute_ranking_difference(p1["rank"], p2["rank"]),
        "winner_rank_points": p1["rank_points"],
        "loser_rank_points": p2["rank_points"],
        "rank_points_diff": p1["rank_points"] - p2["rank_points"],
        "tourney_tier": default_tier,
        "winner_double_fault_rate": p1["double_fault_rate"],
        "loser_double_fault_rate": p2["double_fault_rate"],
        "double_fault_diff": p2["double_fault_rate"] - p1["double_fault_rate"],
        "winner_days_since_last": p1["days_since_last"],
        "loser_days_since_last": p2["days_since_last"],
        "days_since_last_diff": p1["days_since_last"] - p2["days_since_last"],
        "winner_h2h": 0.5,
        "loser_h2h": 0.5,
        "h2h_diff": 0.0,
    }])
    return live_X


def main():
    parser = argparse.ArgumentParser(description="Predict a live ATP/WTA match using Project ACE.")
    parser.add_argument("--player1", type=str, required=True, help="Name of Player A")
    parser.add_argument("--player2", type=str, required=True, help="Name of Player B")
    parser.add_argument("--surface", type=str, required=True, help="Surface (Hard, Clay, Grass)")
    args = parser.parse_args()

    print(f"\n--- Project ACE Prediction Engine ---")
    print(f"Matchup: {args.player1} vs {args.player2} ({args.surface})")

    df, X, y, final_ratings = load_and_prepare_data()
    print(f"Loaded {len(X)} feature rows for training.")

    print("Training XGBoost model on real historical features...")
    model = xgb.XGBClassifier(
        max_depth=4, learning_rate=0.05, n_estimators=300,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=4,
        verbosity=0, use_label_encoder=False
    )
    model.fit(X.drop(columns=["year"]), y)

    live_features = extract_live_features(args.player1, args.player2, args.surface, df, final_ratings)
    raw_probs = model.predict_proba(live_features)[0]

    print("\n===============================")
    print(f"FINAL MODEL PROBABILITIES")
    print("===============================")
    print(f"2-0 ({args.player1} sweeps):  {raw_probs[0]:.4f}")
    print(f"2-1 ({args.player1} in 3):    {raw_probs[1]:.4f}")
    print(f"0-2 ({args.player2} sweeps):  {raw_probs[2]:.4f}")
    print(f"1-2 ({args.player2} in 3):    {raw_probs[3]:.4f}")
    print("===============================")
    print("You can now copy/paste these probabilities into 'manual_ingest.py' to search the market!\n")

if __name__ == "__main__":
    main()
