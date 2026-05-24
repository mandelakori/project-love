"""
demo_bet.py — Run a real Project ACE betting scenario using actual ATP historical data.

This script loads the real dataset from data/, trains the live model, computes
set-score probabilities for a match, and evaluates actual sportsbook odds for +EV dutching.
"""

import argparse

import pandas as pd
import xgboost as xgb
from dutcher import derive_market_probs, find_optimal_dutch

from run_ace import (
    load_atp_data,
    load_ranking_data,
    enrich_match_ranks,
    parse_target,
    compute_elo_ratings,
    build_features,
    get_player_live_features,
    compute_tournament_tier,
    compute_ranking_difference,
    normalize_key,
)


def prepare_training_data():
    print("Loading real ATP match history from data/...")
    df = load_atp_data()
    ranking_df = load_ranking_data()

    df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce")
    df = df.sort_values("tourney_date").reset_index(drop=True)
    df = enrich_match_ranks(df, ranking_df)
    df["year"] = (df["tourney_date"] // 10000).astype(int)
    df["target"] = df.apply(lambda r: parse_target(r.get("score", ""), r.get("best_of", 3)), axis=1)

    df, final_ratings = compute_elo_ratings(df)
    X, y = build_features(df[df["target"] >= 0])
    return df, final_ratings, X, y


def train_model(X, y):
    model = xgb.XGBClassifier(
        max_depth=4,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=4,
        verbosity=0,
        use_label_encoder=False,
    )
    model.fit(X.drop(columns=["year"]), y)
    return model


def format_probability_line(p1, p2, probs):
    print("\n===============================")
    print("FINAL MODEL PROBABILITIES")
    print("===============================")
    print(f"2-0 ({p1} sweeps):  {probs[0]:.4f}")
    print(f"2-1 ({p1} in 3):    {probs[1]:.4f}")
    print(f"0-2 ({p2} sweeps):  {probs[2]:.4f}")
    print(f"1-2 ({p2} in 3):    {probs[3]:.4f}")
    print("===============================")


def main():
    parser = argparse.ArgumentParser(description="Run a live Project ACE betting scenario with real data.")
    parser.add_argument("--player1", type=str, help="Name of Player 1")
    parser.add_argument("--player2", type=str, help="Name of Player 2")
    parser.add_argument("--surface", type=str, default="Hard", help="Surface (Hard/Clay/Grass)")
    args = parser.parse_args()

    p1 = args.player1 or input("Enter Player 1 (e.g., Andrey Rublev): ").strip()
    p2 = args.player2 or input("Enter Player 2 (e.g., Hamad Medjedovic): ").strip()
    surface = args.surface or input("Enter Surface (Hard/Clay/Grass): ").strip()

    df, final_ratings, X, y = prepare_training_data()
    print(f"Loaded {len(X)} labeled feature rows for model training.")

    print("Training the live XGBoost model on actual ATP history...")
    model = train_model(X, y)

    p1_features = get_player_live_features(df, p1, final_ratings)
    p2_features = get_player_live_features(df, p2, final_ratings)
    tourney_tier = compute_tournament_tier(None)

    live_X = pd.DataFrame([
        {
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
            "tourney_tier": tourney_tier,
            "winner_double_fault_rate": p1_features["double_fault_rate"],
            "loser_double_fault_rate": p2_features["double_fault_rate"],
            "double_fault_diff": p2_features["double_fault_rate"] - p1_features["double_fault_rate"],
            "winner_days_since_last": p1_features["days_since_last"],
            "loser_days_since_last": p2_features["days_since_last"],
            "days_since_last_diff": p1_features["days_since_last"] - p2_features["days_since_last"],
            "winner_h2h": 0.5,
            "loser_h2h": 0.5,
            "h2h_diff": 0.0,
            "year": 0,
        }
    ])

    raw_probs = model.predict_proba(live_X)[0]
    format_probability_line(p1, p2, raw_probs)

    core_probs = {"2-0": raw_probs[0], "2-1": raw_probs[1], "0-2": raw_probs[2], "1-2": raw_probs[3]}
    derived_probs = derive_market_probs(core_probs)

    print("\nSupported markets:")
    for key in sorted(derived_probs):
        print(f"  - {key.replace('Player 1', p1).replace('Player 2', p2)}")

    while True:
        odds_inp = input("\nEnter sportsbook odds in the format Market=Odds, Market=Odds (or 'q' to quit)> ").strip()
        if odds_inp.lower() == "q":
            break

        raw_odds = {}
        for pair in odds_inp.split(","):
            try:
                k, v = pair.rsplit("=", 1)
                raw_odds[k.strip()] = float(v.strip())
            except ValueError:
                print(f"Could not parse market entry: {pair.strip()}")

        m_odds = {}
        skipped = []
        for key, val in raw_odds.items():
            normalized = normalize_key(key, p1, p2)
            if normalized in derived_probs:
                m_odds[normalized] = val
            else:
                skipped.append(key)

        if skipped:
            print(f"Skipped unsupported markets: {', '.join(skipped)}")
        if not m_odds:
            print("No valid sportsbook markets were recognized. Please match the supported market names above.")
            continue

        res = find_optimal_dutch(derived_probs, m_odds, bankroll=100, kelly_fraction=0.25)
        print("\n==== OPTIMAL DUTCHING RESULT ====")
        if res["decision"] == "BET":
            legs = [leg.replace("Player 1", p1).replace("Player 2", p2) for leg in res["covered_legs"]]
            print(f"[BET] EDGE FOUND! (+{res['edge']:.4f} vs the book)")
            print(f"Optimal legs: {legs}")
            print("Recommended stakes:")
            for key, stake in res["stakes"].items():
                print(f"  - {key.replace('Player 1', p1).replace('Player 2', p2)}: ${stake:.2f}")
            print(f"Guaranteed return: ${res['guaranteed_return']:.2f}")
        else:
            print("[NO BET] No +EV combination found for these input odds.")
        print("==============================")

if __name__ == "__main__":
    main()
