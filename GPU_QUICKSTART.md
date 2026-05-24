# Quick Reference: Running on GPU Machine

## Copy-Paste Commands

### 1. Clone Repository (on GPU machine)
```bash
# Replace <your-github-url> with your actual repo URL
git clone <your-github-url> project6_sports_betting
cd project6_sports_betting
```

### 2. Set Up Environment
```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install CPU dependencies
pip install -r requirements.txt

# Install GPU XGBoost (NVIDIA CUDA required)
pip install xgboost-gpu
```

### 3. Download Data
```bash
python ingest_data.py
```

(One-time: ~5-10 min, ~2GB downloaded)

### 4. Run Optuna Search on GPU

**Quick (20 trials, 2 workers, ~20-30 min):**
```bash
python train_gpu.py --n-trials 20 --n-workers 2
```

**Recommended (50 trials, 4 workers, ~1-2 hours):**
```bash
python train_gpu.py --n-trials 50 --n-workers 4
```

**Large (100 trials, 8 workers, ~3-5 hours):**
```bash
python train_gpu.py --n-trials 100 --n-workers 8
```

**With verbose output (add `--no-retrain` to skip final retraining):**
```bash
python train_gpu.py --n-trials 50 --n-workers 4 --no-retrain
```

### 5. Use Trained Model
```bash
python run_ace.py
```

Prompts for player names and live odds. Returns optimal stakes and predictions.

---

## Expected Times (NVIDIA RTX 3090 / A100 equivalent)

| Option | Time |
|--------|------|
| 20 trials, 2 workers | ~20-30 min |
| 50 trials, 4 workers | ~1-2 hours |
| 100 trials, 8 workers | ~3-5 hours |

**Note**: Actual time depends on GPU, number of trials, CPU cores, and data size.

---

## Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# If not found, install NVIDIA drivers for your GPU

# Verify XGBoost GPU:
python -c "import xgboost; print(xgboost.__version__); xgboost.build_info()"
```

### Out of Memory
- Reduce `--n-workers` (e.g., 2 instead of 4)
- Run on fewer trials first (e.g., 20 instead of 50)

### Data Not Found
```bash
python ingest_data.py
# Re-downloads ATP match CSVs into data/
```

---

## Output Files

- `models/best_optuna_model.pkl` — Best trained model (auto-saved after Optuna completes)
- `.optuna/` — Optuna study database (for resuming interrupted searches)
- `logs/` — Training logs (if created)

---

## Next: Push to GitHub

```bash
# If you haven't set up a GitHub repo yet:
git remote add origin <your-github-url>
git push -u origin master

# Later, commit your GPU results:
git add models/best_optuna_model.pkl
git commit -m "Optuna GPU training: 50 trials, best log-loss X.XXX"
git push origin master
```

---

## Questions?

Refer to `README.md` for full documentation.
