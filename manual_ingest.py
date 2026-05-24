import json
import argparse
import time
from dutcher import derive_market_probs, find_optimal_dutch

def watch_json(file_path):
    print(f"Watching {file_path} for dynamic market odds...")
    while True:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if "core_probs" in data and "market_odds" in data:
                    probs = derive_market_probs(data["core_probs"])
                    res = find_optimal_dutch(probs, data["market_odds"], bankroll=100)
                    print(f"\n[{time.time()}] Dynamic Scan Result: {res['decision']} (Edge: {res.get('edge', 0)})")
        except Exception:
            pass
        time.sleep(5)

def main():
    parser = argparse.ArgumentParser(description="Dynamic Market Ingest utility for Project ACE")
    parser.add_argument('--watch', type=str, help='Path to JSON file to watch')
    args = parser.parse_args()

    if args.watch:
        watch_json(args.watch)
    else:
        print("Interactive Dynamic Market Scanner")
        while True:
            try:
                print("\n1. Enter core probabilities for: 2-0, 2-1, 0-2, 1-2")
                prob_inp = input("Probs > ")
                if prob_inp.lower() == 'q': break
                
                print("\n2. Enter available odds (Format: key=value, key=value)")
                print("Valid keys based on website: 'Match winner - Player 1', 'Total sets - over 2.5', 'Correct score - 2:0', etc.")
                odds_inp = input("Odds > ")
                if odds_inp.lower() == 'q': break
                
                p_vals = [float(x.strip()) for x in prob_inp.split(',')]
                core_p = {"2-0": p_vals[0], "2-1": p_vals[1], "0-2": p_vals[2], "1-2": p_vals[3]}
                
                m_odds = {}
                for pair in odds_inp.split(','):
                    k, v = pair.split('=')
                    m_odds[k.strip()] = float(v.strip())
                    
                derived_probs = derive_market_probs(core_p)
                res = find_optimal_dutch(derived_probs, m_odds, bankroll=100, kelly_fraction=0.25)
                
                print("\n==== OPTIMAL COMBINATION FOUND ====")
                if res["decision"] == "BET":
                    print(f"Status: EDGE FOUND! (+{res['edge']:.4f})")
                    print(f"Optimal Legs to Bet: {res['covered_legs']}")
                    print(f"Recommended Stakes (from 100 Bankroll):")
                    for k, v in res['stakes'].items():
                        print(f"  - {k}: ${v:.2f}")
                else:
                    print("NO BET - No profitable combination yields +EV across these markets.")
                print("===================================\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print("Input error:", e)

if __name__ == '__main__':
    main()
