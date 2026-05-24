# Dutcher logic

def dutch(model_probs, book_odds, bankroll, kelly_fraction=0.25):
    legs = list(model_probs.keys())

    # Step 1: Total implied probability from the book
    book_implied = sum(1 / book_odds[k] for k in legs)

    # Step 2: Total model confidence on covered outcomes
    model_total = sum(model_probs[k] for k in legs)

    # Step 3: +EV gate — only proceed if model beats the book
    if model_total <= book_implied:
        return {"decision": "NO BET", "edge": model_total - book_implied}

    edge = model_total - book_implied

    # Step 4: Kelly stake on the Dutch as a whole
    # Kelly for a Dutch: f = edge / (1 - book_implied)
    kelly_stake = (edge / (1 - book_implied)) * kelly_fraction * bankroll

    # Step 5: Allocate stake proportionally to 1/odds (equal-profit Dutch)
    weights = {k: (1 / book_odds[k]) / book_implied for k in legs}
    stakes  = {k: kelly_stake * weights[k] for k in legs}

    return {
        "decision": "BET",
        "edge": round(edge, 4),
        "kelly_stake": round(kelly_stake, 2),
        "stakes": {k: round(v, 2) for k, v in stakes.items()},
        "guaranteed_return": round(kelly_stake / book_implied, 2),
    }

import itertools

def derive_market_probs(core_probs):
    """
    Transforms the 4 base `{2-0, 2-1, 0-2, 1-2}` probabilities into aggregated markets.
    Using exact website categories.
    """
    markets = {}
    
    # Correct scores
    markets["Correct score - 2:0"] = core_probs.get("2-0", 0)
    markets["Correct score - 2:1"] = core_probs.get("2-1", 0)
    markets["Correct score - 0:2"] = core_probs.get("0-2", 0)
    markets["Correct score - 1:2"] = core_probs.get("1-2", 0)
    
    # Match Winner
    markets["Match winner - Player 1"] = core_probs.get("2-0", 0) + core_probs.get("2-1", 0)
    markets["Match winner - Player 2"] = core_probs.get("0-2", 0) + core_probs.get("1-2", 0)
    
    # Total Sets
    markets["Total sets - over 2.5"] = core_probs.get("2-1", 0) + core_probs.get("1-2", 0)
    markets["Total sets - under 2.5"] = core_probs.get("2-0", 0) + core_probs.get("0-2", 0)
    
    # Set Handicaps (-1.5 and +1.5)
    markets["Set handicap - (-1.5) Player 1"] = core_probs.get("2-0", 0)
    markets["Set handicap - (1.5) Player 1"] = core_probs.get("2-0", 0) + core_probs.get("2-1", 0) + core_probs.get("1-2", 0)
    markets["Set handicap - (-1.5) Player 2"] = core_probs.get("0-2", 0)
    markets["Set handicap - (1.5) Player 2"] = core_probs.get("0-2", 0) + core_probs.get("1-2", 0) + core_probs.get("2-1", 0)

    # To win a set
    markets["Player 1 to win a set - yes"] = markets["Set handicap - (1.5) Player 1"]
    markets["Player 2 to win a set - yes"] = markets["Set handicap - (1.5) Player 2"]
    
    return markets

def find_optimal_dutch(prob_dict, odds_dict, bankroll, kelly_fraction=0.25):
    """
    Searches all combinatoric arrangements to find the highest +EV Dutching edge.
    """
    # Only search over legs where we have odds
    available_legs = list(odds_dict.keys())
    
    best_edge = 0
    best_dutch = {"decision": "NO BET", "edge": 0}
    
    # Try all combinations of 1 to N legs
    for r in range(1, len(available_legs) + 1):
        for combo in itertools.combinations(available_legs, r):
            c_probs = {k: prob_dict[k] for k in combo if k in prob_dict}
            c_odds = {k: odds_dict[k] for k in combo}
            
            # Must have probability for all odds in the combo
            if len(c_probs) == len(c_odds) and len(c_probs) > 0:
                res = dutch(c_probs, c_odds, bankroll, kelly_fraction)
                if res["decision"] == "BET" and res["edge"] > best_edge:
                    best_edge = res["edge"]
                    best_dutch = res
                    best_dutch["covered_legs"] = list(combo)
                    
    return best_dutch
