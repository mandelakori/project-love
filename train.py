import pandas as pd
import optuna
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import numpy as np

from run_ace import load_atp_data, load_ranking_data, enrich_match_ranks, build_features, parse_target, evaluate_time_series_cv

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators":     trial.suggest_int("n_estimators", 200, 1500),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda":           trial.suggest_float("lambda", 0.1, 10.0, log=True),
        "alpha":            trial.suggest_float("alpha", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "tree_method":      "gpu_hist",  # assuming gpu available as per spec
        "device":           "cuda",
        "objective":        "multi:softprob",
        "num_class":        4,  # {2-0, 2-1, 0-2, 1-2}
    }

    from sklearn.model_selection import train_test_split
    # Use 20% of the training data purely for calibration
    X_train_sub, X_calib, y_train_sub, y_calib = train_test_split(
        X_train, y_train, test_size=0.2, shuffle=False
    )

    try:
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_sub, y_train_sub)
    except xgb.core.XGBoostError as e:
        if "gpu_hist" in str(e) or "cuda" in str(e):
            print("CUDA not available. Falling back to CPU/hist.")
            params["tree_method"] = "hist"
            params["device"] = "cpu"
            model = xgb.XGBClassifier(**params)
            model.fit(X_train_sub, y_train_sub)
        else:
            raise e

    # Raw predictions
    val_probs = model.predict_proba(X_val)
    calib_probs = model.predict_proba(X_calib)

    calibrated_val_probs = val_probs
    if len(np.unique(y_calib)) > 1:
        try:
            lr = LogisticRegression(multi_class='multinomial', max_iter=1000)
            lr.fit(calib_probs, y_calib)
            calibrated_val_probs = lr.predict_proba(val_probs)
        except Exception:
            calibrated_val_probs = val_probs

    ll = log_loss(y_val, calibrated_val_probs, labels=[0, 1, 2, 3])
    return ll

def optimize_model(df):
    """
    df: DataFrame containing features, 'target' (0,1,2,3), and 'year'
    for time-series cross validation.
    """
    years = sorted(df['year'].unique())
    
    def objective_wrapper(trial):
        log_losses = []
        # Time-series CV boundaries: always train on past, validate on future.
        for i in range(1, len(years)):
            train_years = years[:i]
            val_year = years[i]
            
            train_mask = df['year'].isin(train_years)
            val_mask = df['year'] == val_year
            
            X_train = df[train_mask].drop(columns=['target', 'year'])
            y_train = df[train_mask]['target']
            
            X_val = df[val_mask].drop(columns=['target', 'year'])
            y_val = df[val_mask]['target']
            
            # Require at least some training samples
            if len(X_train) < 50 or len(X_val) == 0:
                continue
                
            ll = objective(trial, X_train, y_train, X_val, y_val)
            if np.isfinite(ll):
                log_losses.append(ll)
    study.optimize(objective_wrapper, n_trials=50)
    
    print("Best params found by Optuna:", study.best_params)
    return study.best_params


def load_training_data():
    df = load_atp_data()
    ranking_df = load_ranking_data()
    df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce")
    df = df.sort_values("tourney_date").reset_index(drop=True)
    df = enrich_match_ranks(df, ranking_df)
    df = df.dropna(subset=["tourney_date"]).copy()
    df["year"] = (df["tourney_date"] // 10000).astype(int)
    df["target"] = df.apply(lambda r: parse_target(r.get("score", ""), r.get("best_of", 3)), axis=1)
    X, y = build_features(df[df["target"] >= 0])
    return X, y


def main():
    X, y = load_training_data()
    print(f"Loaded {len(X)} feature rows from real ATP data.")
    cv_loss = evaluate_time_series_cv(X, y)
    if not np.isnan(cv_loss):
        print(f"Time-series CV log loss: {cv_loss:.4f}")
    else:
        print("Warning: not enough distinct years for time-series validation.")
    print("Running Optuna hyperparameter search on real features...")
    df_train = X.copy()
    df_train["year"] = X["year"]
    df_train["target"] = y
    best_params = optimize_model(df_train)
    print("Done. Best params:", best_params)


if __name__ == '__main__':
    main()
