# Project ACE – Tennis Betting Pipeline with GPU Hyperparameter Optimization

A unified, production-ready end-to-end sports betting prediction system for ATP tennis matches.

## Features

- **Real Data**: Ingests Sackmann ATP CSVs (1968–2026) with ranking enrichment via `merge_asof`
- **Feature Engineering**: Elo ratings, SDI, double-fault rates, days-since-last-match, head-to-head (H2H), tournament tier, ranking/ranking-points
- **Time-Series CV**: Year-based cross-validation for robust out-of-sample evaluation
- **Live Inference**: Extracts current player features and makes real-time predictions
- **Optuna Hyperparameter Search**: GPU-accelerated Bayesian optimization with early stopping
- **Dutching**: Finds +EV multi-leg betting opportunities and computes optimal stake allocation
- **Model Persistence**: Saves best models to `models/final_model.pkl`

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support (NVIDIA CUDA):
```bash
pip install xgboost-gpu
```

### 2. Download Data

```bash
python ingest_data.py
```

This populates `data/` with ATP match CSVs and ranking snapshots.

### 3. Train a Model

**Option A: Quick Training (CPU, 2–5 min)**
```bash
python -c "
import run_ace, xgboost as xgb, pickle, os
df, X, y = run_ace.load_training_data()
X_feat = X.drop(columns=['year']) if 'year' in X.columns else X
model = xgb.XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=300, 
                          verbosity=0, use_label_encoder=False, objective='multi:softprob')
model.fit(X_feat, y)
os.makedirs('models', exist_ok=True)
pickle.dump(model, open('models/final_model.pkl', 'wb'))
print('✓ Model saved to models/final_model.pkl')
"
```

**Option B: GPU Optuna Hyperparameter Search (GPU, ~1–2 hours for 50 trials)**
```bash
python train_gpu.py --n-trials 50 --n-workers 4
```

### 4. Live Prediction

```bash
python run_ace.py
```

Prompts for player names, surface, and live sportsbook odds. Returns:
- AI-predicted set-score probabilities (2-0, 2-1, 0-2, 1-2)
- Derived match-winner, set-count, and exacta probabilities
- Optimal Dutching stakes for +EV opportunities

## Architecture

### Core Modules

- **`run_ace.py`**: Unified engine (load data, enrich, compute Elo/features, evaluate CV, live predict)
  - `load_atp_data()`: Loads all Sackmann ATP match CSVs
  - `load_ranking_data()`: Loads ATP ranking snapshots
  - `enrich_match_ranks()`: Merges rankings by player & date via `merge_asof`
  - `compute_elo_ratings()`: Tracks rolling Elo for each player
  - `build_features()`: Computes all match features (Elo diff, SDI diff, rank diff, H2H, etc.)
  - `get_player_live_features()`: Extracts latest player stats
  - `compute_live_h2h()`: Computes head-to-head record from historical matches
  - `evaluate_time_series_cv()`: Time-series year-based CV evaluation
  - `main()`: Interactive betting loop

- **`train.py`**: Optuna hyperparameter search (CPU version)
  - `objective(trial, ...)`: Defines hyperparameter space & objective (multi:softprob log-loss)
  - `optimize_model(df)`: Runs time-series CV across all years
  - Fallback to CPU if GPU unavailable

- **`train_gpu.py`**: GPU-optimized Optuna search with parallelization
  - Auto-detects CUDA and configures XGBoost GPU backend
  - Parallel trial execution
  - Early stopping & pruning
  - Calibration via Platt scaling
  - Best model saved to `models/best_optuna_model.pkl`

- **`feature_builder.py`**: Feature computation helpers
  - `compute_double_fault_rate()`, `compute_days_since_last_match()`, `compute_tournament_tier()`, etc.

- **`dutcher.py`**: Dutch/aribtrage betting optimizer
  - `derive_market_probs()`: Converts set-score probs to match-winner, set-count, etc.
  - `find_optimal_dutch()`: Kelly-fraction stake optimization for +EV

- **`predict_match.py`**: CLI for single-match prediction (legacy; use `run_ace.py` instead)

- **`ingest_data.py`**: Downloads fresh Sackmann ATP CSVs into `data/`

### Data Flow

```
Sackmann CSVs (data/)
    ↓
load_atp_data() + load_ranking_data()
    ↓
enrich_match_ranks() [merge_asof by player/date]
    ↓
compute_elo_ratings() [rolling Elo]
    ↓
build_features() [22+ numeric features]
    ↓
evaluate_time_series_cv() [year-by-year splits]
    ↓
Train XGBoost (CPU or GPU)
    ↓
get_player_live_features() + compute_live_h2h()
    ↓
predict_proba() → derive_market_probs()
    ↓
find_optimal_dutch() → stakes
```

## Training Details

### Hyperparameter Search Space (Optuna)

- `max_depth`: [3, 8]
- `learning_rate`: [0.01, 0.3] log-uniform
- `n_estimators`: [200, 1500]
- `subsample`: [0.5, 1.0]
- `colsample_bytree`: [0.5, 1.0]
- `lambda` (L2): [0.1, 10.0] log-uniform
- `alpha` (L1): [0.0, 5.0]
- `min_child_weight`: [1, 10]

### Objective

Minimize cross-validated log-loss on a 4-class target (2-0 / 2-1 / 0-2 / 1-2 set scores).

### Cross-Validation Strategy

**Time-series**: For each year `i`, train on years 1..i-1, validate on year `i`.
- Avoids look-ahead bias
- Respects temporal ordering
- ~25 folds (1995–2024)

### GPU Setup (Recommended)

```bash
# Check CUDA availability
nvidia-smi

# Install XGBoost with GPU support
pip install xgboost-gpu

# On WSL2 with NVIDIA GPU:
# Ensure NVIDIA Container Toolkit is installed
```

## Running on GPU Machine

1. **Clone/push repo to GPU machine:**
   ```bash
   git clone <your-repo-url> project6_sports_betting
   cd project6_sports_betting
   ```

2. **Install deps (GPU environment):**
   ```bash
   pip install -r requirements.txt
   pip install xgboost-gpu  # GPU XGBoost
   ```

3. **Download data:**
   ```bash
   python ingest_data.py
   ```

4. **Run Optuna search with GPU:**
   ```bash
   # Quick: 20 trials, 2 workers
   python train_gpu.py --n-trials 20 --n-workers 2
   
   # Medium: 50 trials, 4 workers (recommended)
   python train_gpu.py --n-trials 50 --n-workers 4
   
   # Large: 100 trials, 8 workers (if you have 16+ CPU cores + plenty of RAM)
   python train_gpu.py --n-trials 100 --n-workers 8
   ```

5. **Monitor progress:**
   - Optuna prints trial results and best params in real-time
   - Best model auto-saved to `models/best_optuna_model.pkl`

6. **Use trained model for live prediction:**
   ```bash
   python run_ace.py
   ```

## Model Performance

- **Time-Series CV Log-Loss**: ~1.15–1.35 (depends on hyperparams; lower is better)
- **Training Data**: 706k+ valid match records
- **Feature Count**: 22 (Elo diff, rank diff, H2H, SDI diff, tournament tier, double-fault rate, recency, etc.)
- **Training Time (CPU, n_estimators=300)**: ~2–5 minutes
- **Training Time (GPU, n_estimators=300)**: ~30–60 seconds
- **Optuna Search Time (GPU, 50 trials, 4 workers)**: ~30–90 minutes

## Testing & Validation

```bash
# Syntax check
python -m py_compile run_ace.py train.py train_gpu.py feature_builder.py dutcher.py

# Quick smoke test (fits small model)
python -c "
from train import load_training_data
import xgboost as xgb
X, y = load_training_data()
model = xgb.XGBClassifier(max_depth=3, n_estimators=10, verbosity=0)
model.fit(X.head(1000).drop(columns=['year']), y[:1000])
print('✓ Smoke test passed')
"
```

## Dependencies

See `requirements.txt` for full list. Key packages:
- `pandas`, `numpy`: Data processing
- `xgboost`: ML model
- `scikit-learn`: Metrics, calibration
- `optuna`: Hyperparameter optimization
- `scipy`: Misc math

## Troubleshooting

### XGBoost GPU not detected
- Ensure NVIDIA drivers and CUDA toolkit installed
- Install `xgboost-gpu` explicitly: `pip install xgboost-gpu`
- Fall back to CPU: Remove `tree_method="gpu_hist"` and `device="cuda"` from code

### Out of Memory (OOM)
- Reduce `n_workers` in `train_gpu.py`
- Reduce `n_estimators` in trial space
- Use a smaller subset of data for prototyping

### Ranking data not loading
- Run `python ingest_data.py` to re-download
- Check that `data/atp_rankings_*.csv` files exist

## Next Steps

1. **Deploy**: Copy best model + `run_ace.py` to production
2. **Real betting**: Integrate with sportsbook APIs
3. **Live monitoring**: Track prediction accuracy & ROI over time
4. **Retraining**: Re-run Optuna monthly as new match data arrives

## License

Internal use only.
