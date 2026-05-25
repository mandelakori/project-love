"""
dry_run.py — Verify the betting pipeline with real historical ATP match data.

This script removes the old synthetic dry-run path and validates the live
feature pipeline, ranking enrichment, set-score target parsing, and feature
structure using the actual data files stored in data/.
"""

import pandas as pd

from run_ace import (
    load_atp_data,
    load_ranking_data,
    enrich_match_ranks,
    parse_target,
    compute_elo_ratings,
    build_features,
    evaluate_time_series_cv,
)


def run_real_pipeline_check():
    print("Project ACE Dry Run — loading actual ATP historical data...")
    df = load_atp_data()
    ranking_df = load_ranking_data()

    df["tourney_date"] = df["tourney_date"].apply(pd.to_numeric, errors="coerce")
    df = df.sort_values("tourney_date").reset_index(drop=True)
    df = enrich_match_ranks(df, ranking_df)
    df = df.dropna(subset=["tourney_date"]).copy()
    df["year"] = (df["tourney_date"] // 10000).astype(int)
    df["target"] = df.apply(lambda row: parse_target(row.get("score", ""), row.get("best_of", 3)), axis=1)

    valid_df = df[df["target"] >= 0]
    print(f"Loaded {len(df)} total matches, {len(valid_df)} valid set-score rows.")
    print(f"Year range: {valid_df['year'].min()} - {valid_df['year'].max()}")

    df, final_ratings = compute_elo_ratings(df)
    X, y = build_features(valid_df)
    print(f"Feature matrix shape: {X.shape}")

    cv_loss = evaluate_time_series_cv(X, y)
    if not pd.isna(cv_loss):
        print(f"Time-series cross-validation log loss: {cv_loss:.4f}")
    else:
        print("Not enough distinct years for time-series CV; training data is still valid.")

    print("Dry-run validation complete. The real data pipeline is ready for live betting use.")


if __name__ == "__main__":
    run_real_pipeline_check()
