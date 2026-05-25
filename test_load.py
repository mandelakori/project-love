import pandas as pd, os, importlib.util

# Load run_ace module
spec = importlib.util.spec_from_file_location("run_ace", os.path.abspath("run_ace.py"))
run_ace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_ace)

# Load data
df = run_ace.load_atp_data()
ranking_df = run_ace.load_ranking_data()

# Convert tourney_date to numeric and drop NaNs
df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce")
df = df.dropna(subset=["tourney_date"]).reset_index(drop=True)
print("Rows after dropna:", len(df))
print("NaNs in tourney_date after dropna:", df["tourney_date"].isna().sum())

# Compute year with nullable integer dtype
df["year"] = (df["tourney_date"] // 10000).astype("Int64")
print("Year NA count:", df["year"].isna().sum())
