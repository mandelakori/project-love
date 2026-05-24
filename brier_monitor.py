import pandas as pd
import numpy as np

def check_brier(ledger_path, window=50, threshold=0.25):
    try:
        df = pd.read_csv(ledger_path)
    except FileNotFoundError:
        return 0.0

    if len(df) == 0:
        return 0.0

    df = df.tail(window)
    # Brier Score = 1/N * sum((predicted_prob - actual_outcome)^2)
    # The specification assumes the dataframe has a column `predicted_prob` corresponding to 
    # the probability of the outcome that ACTUALLY occurred.
    if 'predicted_prob' in df.columns:
        bs = ((df["predicted_prob"] - 1.0) ** 2).mean()
        if bs > threshold:
            raise SystemExit(
                f"[ACE HALT] Brier Score {bs:.4f} exceeds {threshold}. "
                "Model is underperforming market. Retrain before resuming."
            )
        return bs
    return 0.0

def check_ece(ledger_path, window=50, n_bins=20, threshold=0.15):
    """
    Calculate Expected Calibration Error (ECE) over the same window.
    Bin logic in 0.05 step sizes (n_bins=20 defaults to bins of 0.05).
    """
    try:
        df = pd.read_csv(ledger_path)
    except FileNotFoundError:
        return 0.0

    if len(df) == 0:
        return 0.0

    df = df.tail(window)
    if 'predicted_prob' not in df.columns or 'won' not in df.columns:
        return 0.0
        
    bins = np.linspace(0., 1., n_bins + 1)
    bin_indices = np.digitize(df["predicted_prob"], bins) - 1
    
    ece = 0.0
    total_samples = len(df)
    
    for i in range(n_bins):
        bin_mask = bin_indices == i
        if not np.any(bin_mask):
            continue
            
        bin_samples = df[bin_mask]
        # Calculate accuracy for this bin as mean of 'won' column (1 or 0)
        bin_accuracy = bin_samples['won'].mean()
        bin_confidence = bin_samples['predicted_prob'].mean()
        bin_weight = len(bin_samples) / total_samples
        
        ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
        
    if ece > threshold:
         raise SystemExit(
            f"[ACE HALT] Expected Calibration Error (ECE) {ece:.4f} exceeds {threshold}. "
            "Confidence mismatch detected. Review model calibration."
        )
    return ece
