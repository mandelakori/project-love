import pandas as pd
import numpy as np
from elo_engine import SurfaceElo

def compute_sdi(aces, points_played, first_serve_win_pct):
    """
    Service Dominance Index (SDI)
    SDI = (Aces / Points Played) + 1st Serve Win%
    A high SDI predicts the ability to hold serve cheaply.
    """
    if pd.isna(points_played) or points_played == 0:
        return 0.0
    return (aces / points_played) + first_serve_win_pct

def compute_fatigue(matches_df, target_date, decay_lambda=0.15):
    """
    Summation of Minutes * exp(-lambda * (target_date - match_date))
    `matches_df` requires columns: ['match_date', 'minutes']
    """
    target_date = pd.to_datetime(target_date)
    fatigue = 0.0
    for _, row in matches_df.iterrows():
        match_date = pd.to_datetime(row['match_date'])
        days_diff = (target_date - match_date).days
        # only sum fatigue for past matches (where days_diff > 0)
        if days_diff > 0:
            if not pd.isna(row['minutes']):
                fatigue += row['minutes'] * np.exp(-decay_lambda * days_diff)
    return fatigue

def compute_break_point_conversion(bp_saved, bp_faced):
    """Break point hold rate: bp_saved / bp_faced (defense)"""
    if pd.isna(bp_faced) or bp_faced == 0:
        return 0.5
    return bp_saved / bp_faced

def compute_first_serve_efficiency(first_in, svpt):
    """First serve percentage"""
    if pd.isna(svpt) or svpt == 0:
        return 0.5
    return first_in / svpt

def compute_recent_form(matches_df, target_date, window_days=30):
    """
    Win % in last N days
    `matches_df` needs columns: ['match_date', 'result']
    result: 1 for win, 0 for loss
    """
    target_date = pd.to_datetime(target_date)
    recent = matches_df[
        (pd.to_datetime(matches_df['match_date']) <= target_date) &
        (pd.to_datetime(matches_df['match_date']) > target_date - pd.Timedelta(days=window_days))
    ]
    if len(recent) == 0:
        return 0.5
    return recent['result'].mean()

def compute_surface_win_rate(matches_df, surface):
    """Win % on specific surface"""
    on_surface = matches_df[matches_df['surface'] == surface]
    if len(on_surface) == 0:
        return 0.5
    return on_surface['result'].mean()

def compute_h2h_record(player_id, opponent_id, all_matches):
    """
    Head-to-head record: wins / total
    `all_matches` should have columns: ['winner_id', 'loser_id', 'match_date']
    """
    h2h = all_matches[
        ((all_matches['winner_id'] == player_id) & (all_matches['loser_id'] == opponent_id)) |
        ((all_matches['winner_id'] == opponent_id) & (all_matches['loser_id'] == player_id))
    ]
    
    if len(h2h) == 0:
        return 0.5  # No history
    
    wins = len(h2h[h2h['winner_id'] == player_id])
    return wins / len(h2h)

def compute_service_stats(aces, df, svpt, first_won, second_won, sv_gms):
    """
    Service points won percentage
    """
    if pd.isna(svpt) or svpt == 0:
        return 0.5
    return (first_won + second_won) / svpt

def compute_break_point_conversion_attack(bp_faced, bp_saved):
    """Break point conversion rate (offense): (bp_faced - bp_saved) / bp_faced"""
    if pd.isna(bp_faced) or bp_faced == 0:
        return 0.5
    return (bp_faced - bp_saved) / bp_faced

def compute_double_fault_rate(double_faults, svpt):
    """Double fault rate as a service pressure feature."""
    if pd.isna(svpt) or svpt == 0:
        return 0.05
    if pd.isna(double_faults):
        return 0.05
    return double_faults / svpt

def compute_days_since_last_match(matches_df, target_date):
    """Days since the most recent past match."""
    target_date = pd.to_datetime(target_date)
    prior = pd.to_datetime(matches_df['match_date']) < target_date
    history = matches_df[prior]
    if len(history) == 0:
        return np.nan
    last_match = pd.to_datetime(history['match_date']).max()
    return (target_date - last_match).days

def compute_tournament_tier(tourney_level):
    """Encode tournament level as an ordinal importance feature."""
    if pd.isna(tourney_level):
        return 0
    level = str(tourney_level).strip().upper()
    tier_map = {
        'G': 5,  # Grand Slam
        'A': 4,  # ATP Finals / Masters?
        'M': 4,  # Masters
        'D': 3,  # 500-level / indoor
        'B': 3,
        'C': 2,
        'F': 2,
        'S': 2,
        'P': 1,
        'E': 1,
        'H': 2,
        'X': 2,
    }
    return tier_map.get(level, 0)

def compute_elo_difference(elo_engine, player1_id, player2_id, surface):
    """Get Elo rating difference"""
    elo_engine._ensure_player(player1_id, surface)
    elo_engine._ensure_player(player2_id, surface)
    
    p1_elo = elo_engine.ratings[surface][player1_id]["elo"]
    p2_elo = elo_engine.ratings[surface][player2_id]["elo"]
    
    return p1_elo - p2_elo

def compute_ranking_difference(winner_rank, loser_rank):
    """Rank difference (lower is better)"""
    if pd.isna(winner_rank) or pd.isna(loser_rank):
        return 0
    return loser_rank - winner_rank  # Positive = higher seeded opponent

def compute_win_consistency(matches_df, window_matches=20):
    """
    Standard deviation of wins in rolling window
    Lower = more consistent
    """
    if len(matches_df) < window_matches:
        return matches_df['result'].std() if len(matches_df) > 0 else 0.5
    return matches_df['result'].tail(window_matches).std()
