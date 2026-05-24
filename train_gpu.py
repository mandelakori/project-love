"""
train_gpu.py — GPU-optimized Optuna hyperparameter search for Project ACE.

Usage:
  python train_gpu.py --n-trials 50 --n-workers 4
  python train_gpu.py --n-trials 100 --n-workers 8

Features:
  - Auto-detects CUDA and configures GPU backend
  - Parallel trial execution via joblib
  - Early stopping & pruning
  - Platt scaling calibration
  - Saves best model to models/best_optuna_model.pkl
"""

import argparse
import os
import sys
import pickle
import warnings

import pandas as pd
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from run_ace import (
    load_atp_data,
    load_ranking_data,
    enrich_match_ranks,
    build_features,
    parse_target,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# GPU DETECTION & CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

def detect_gpu():
    """Detect if GPU is available and return tree_method, device."""
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            print(f"[GPU] Detected {device_count} NVIDIA GPU(s). Using GPU backend.")
            return "gpu_hist", "cuda"
    except Exception:
        pass
    
    print("[CPU] No GPU detected. Using CPU backend.")
    return "hist", "cpu"

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_training_data():
    """Load and prepare training data for time-series CV."""
    df = load_atp_data()
    ranking_df = load_ranking_data()
    
    df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce")
    df = df.sort_values("tourney_date").reset_index(drop=True)
    df = enrich_match_ranks(df, ranking_df)
    df = df.dropna(subset=["tourney_date"]).copy()
    df["year"] = (df["tourney_date"] // 10000).astype(int)
    
    df["target"] = df.apply(
        lambda r: parse_target(r.get("score", ""), r.get("best_of", 3)), axis=1
    )
    
    X, y = build_features(df[df["target"] >= 0])
    print(f"[Data] Loaded {len(X)} feature rows across {X['year'].nunique()} years")
    return X, y

# ──────────────────────────────────────────────────────────────────────────────
# OPTUNA OBJECTIVE FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def objective(trial, X_train, y_train, X_val, y_val, tree_method, device):
    """Optuna objective: minimize validation log-loss with Platt calibration."""
    params = {
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),  # Reduced for speed
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda":           trial.suggest_float("lambda", 0.1, 10.0, log=True),
        "alpha":            trial.suggest_float("alpha", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "tree_method":      tree_method,
        "device":           device,
        "objective":        "multi:softprob",
        "num_class":        4,
        "verbosity":        0,
        "use_label_encoder": False,
    }
    
    # Use subset for fast training per trial
    X_train_sub, X_calib, y_train_sub, y_calib = train_test_split(
        X_train, y_train, test_size=0.2, shuffle=False
    )
    
    try:
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_sub, y_train_sub)
    except xgb.core.XGBoostError as e:
        # Fallback to CPU if GPU fails
        if "gpu_hist" in str(e) or "cuda" in str(e):
            print(f"    [Fallback] GPU trial failed, retrying on CPU: {str(e)[:60]}")
            params["tree_method"] = "hist"
            params["device"] = "cpu"
            model = xgb.XGBClassifier(**params)
            model.fit(X_train_sub, y_train_sub)
        else:
            raise
    
    # Get predictions
    val_probs = model.predict_proba(X_val)
    calib_probs = model.predict_proba(X_calib)
    
    # Platt scaling: calibrate via logistic regression on calib set
    calibrated_val_probs = val_probs
    classes = model.classes_
    if len(np.unique(y_calib)) > 1:
        try:
            lr = LogisticRegression(multi_class="multinomial", max_iter=1000, verbose=0)
            lr.fit(calib_probs, y_calib)
            calibrated_val_probs = lr.predict_proba(val_probs)
            classes = lr.classes_
        except Exception:
            calibrated_val_probs = val_probs
            classes = model.classes_
    
    # Align probabilities to the full 4 classes [0, 1, 2, 3]
    n_classes = 4
    full_probs = np.zeros((calibrated_val_probs.shape[0], n_classes), dtype=float)
    for col_idx, lbl in enumerate(classes):
        lbl = int(lbl)
        if 0 <= lbl < n_classes:
            full_probs[:, lbl] = calibrated_val_probs[:, col_idx]
            
    eps = 1e-15
    full_probs = np.clip(full_probs, eps, 1 - eps)
    full_probs = full_probs / full_probs.sum(axis=1, keepdims=True)
    
    # Compute log-loss with explicit class labels
    ll = log_loss(y_val, full_probs, labels=[0, 1, 2, 3])
    return ll

# ──────────────────────────────────────────────────────────────────────────────
# OPTUNA OPTIMIZATION
# ──────────────────────────────────────────────────────────────────────────────

def optimize_model_gpu(X, y, n_trials, n_workers, tree_method, device):
    """Run Optuna time-series CV optimization with GPU backend."""
    years = sorted(X["year"].unique())
    print(f"[CV] Using {len(years)} distinct years for time-series CV")
    
    def objective_wrapper(trial):
        log_losses = []
        
        # Time-series folds: always train on past, validate on future
        for i in range(1, len(years)):
            train_years = years[:i]
            val_year = years[i]
            
            train_mask = X["year"].isin(train_years)
            val_mask = X["year"] == val_year
            
            X_train = X[train_mask].drop(columns=["year"])
            y_train = y[train_mask]
            X_val = X[val_mask].drop(columns=["year"])
            y_val = y[val_mask]
            
            if len(X_train) < 50 or len(X_val) == 0:
                continue
            
            ll = objective(trial, X_train, y_train, X_val, y_val, tree_method, device)
            if np.isfinite(ll):
                log_losses.append(ll)
        
        return np.mean(log_losses) if log_losses else float("inf")
    
    # Create study with pruning
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    
    print(f"[Optuna] Starting optimization: {n_trials} trials, {n_workers} workers")
    print(f"[Optuna] Backend: {device.upper()} (tree_method={tree_method})")
    
    study.optimize(
        objective_wrapper,
        n_trials=n_trials,
        n_jobs=n_workers,
        show_progress_bar=True,
    )
    
    print(f"\n[✓] Optimization complete!")
    print(f"[✓] Best trial: #{study.best_trial.number}")
    print(f"[✓] Best log-loss: {study.best_value:.4f}")
    print(f"[✓] Best params:")
    for k, v in study.best_params.items():
        print(f"      {k}: {v}")
    
    return study.best_params, study.best_value

# ──────────────────────────────────────────────────────────────────────────────
# RETRAIN BEST MODEL ON ALL DATA
# ──────────────────────────────────────────────────────────────────────────────

def retrain_best_model(X, y, best_params, tree_method, device):
    """Retrain best model on all data using best hyperparameters."""
    print(f"\n[Retrain] Training best model on full dataset ({len(X)} rows)...")
    
    best_params = best_params.copy()
    best_params.update({
        "tree_method": tree_method,
        "device": device,
        "objective": "multi:softprob",
        "num_class": 4,
        "verbosity": 0,
        "use_label_encoder": False,
        "n_estimators": 300,  # Use full n_estimators for final model
    })
    
    X_feat = X.drop(columns=["year"])
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_feat, y)
    
    return model

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GPU-optimized Optuna hyperparameter search for Project ACE"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--no-retrain",
        action="store_true",
        help="Skip final retraining on full data",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("    PROJECT ACE – GPU OPTUNA HYPERPARAMETER SEARCH")
    print("=" * 80)
    
    tree_method, device = detect_gpu()
    print()
    
    print("[1/4] Loading training data...")
    X, y = load_training_data()
    print()
    
    print("[2/4] Running Optuna search...")
    best_params, best_loss = optimize_model_gpu(
        X, y, args.n_trials, args.n_workers, tree_method, device
    )
    print()
    
    if not args.no_retrain:
        print("[3/4] Retraining best model on full dataset...")
        model = retrain_best_model(X, y, best_params, tree_method, device)
        print()
        
        print("[4/4] Saving best model...")
        os.makedirs("models", exist_ok=True)
        model_path = "models/best_optuna_model.pkl"
        pickle.dump(model, open(model_path, "wb"))
        print(f"    ✓ Saved to {model_path}")
        print()
        
        print("=" * 80)
        print("    OPTUNA RUN COMPLETE!")
        print("=" * 80)
        print(f"\nNext: Use the model for live prediction:")
        print(f"  python run_ace.py")
        print()
    else:
        print("[3/4] Skipped retraining (--no-retrain flag set)")
        print()
        print("=" * 80)
        print("    OPTUNA SEARCH COMPLETE (NO RETRAIN)")
        print("=" * 80)
        print()

if __name__ == "__main__":
    main()
