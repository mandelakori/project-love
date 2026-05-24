import os
import urllib.request

# Jeff Sackmann's Match Charting Project repository
CHARTING_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/"
CHARTING_FILES = [
    "charting-m-matches.csv",
    "charting-m-points-2010s.csv",
    "charting-m-points-2020s.csv"
]

# Jeff Sackmann's standard ATP matches for complete metadata
ATP_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/"
ATP_FILES = (
    [f"atp_matches_{year}.csv" for year in range(1968, 2027)] +  # Main matches 1968-2026
    ["atp_matches_amateur.csv"] +  # Pre-Open Era
    [f"atp_matches_doubles_{year}.csv" for year in range(2000, 2021)] +  # Doubles 2000-2020
    [f"atp_matches_futures_{year}.csv" for year in range(1991, 2027)] +  # Futures 1991-2026
    [f"atp_matches_qual_chall_{year}.csv" for year in range(1978, 2027)] +  # Qual/Chall 1978-2026
    ["atp_players.csv"] +  # Player data
    ["atp_rankings_70s.csv", "atp_rankings_80s.csv", "atp_rankings_90s.csv", 
     "atp_rankings_00s.csv", "atp_rankings_10s.csv", "atp_rankings_20s.csv",
     "atp_rankings_current.csv"]  # Rankings
)

def download_file(base_url, file_name, target_dir):
    target_path = os.path.join(target_dir, file_name)
    if not os.path.exists(target_path):
        print(f"Downloading {file_name}...")
        try:
            url = base_url + file_name
            urllib.request.urlretrieve(url, target_path)
            print(f"Saved to {target_path}")
        except Exception as e:
            print(f"Failed to download {file_name}: {e}")
    else:
        print(f"{file_name} already exists. Skipping.")

def ingest_data(target_dir="data"):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print("Fetching Charting Data...")
    for file_name in CHARTING_FILES:
        download_file(CHARTING_URL, file_name, target_dir)

    print("\Fetching ATP Match Data...")
    for file_name in ATP_FILES:
        download_file(ATP_URL, file_name, target_dir)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    ingest_data(data_dir)
