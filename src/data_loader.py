"""
CausTab — Data Loader
Loads nhanes_master.csv and prepares it for modelling.

What this file does:
- Loads the clean NHANES dataset
- Splits features (X), outcome (Y), and environments (E)
- Standardizes features so all variables are on the same scale
- Splits data into train and test sets
- Returns data organized by environment — critical for CausTab
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# ── Column definitions ─────────────────────────────────────────────────────────
FEATURE_COLS = [
    'RIDAGEYR',   # Age
    'RIAGENDR',   # Gender
    'RIDRETH3',   # Race/ethnicity
    'INDFMPIR',   # Income-to-poverty ratio
    'DMDEDUC2',   # Education level
    'BPXSY1',     # Systolic BP reading 1
    'BPXDI1',     # Diastolic BP reading 1
    'BPXSY2',     # Systolic BP reading 2
    'BPXDI2',     # Diastolic BP reading 2
    'BMXBMI',     # BMI
    'BMXWAIST',   # Waist circumference
]

OUTCOME_COL = 'hypertension'
ENV_COL     = 'environment'
ENV_ORDER   = ['2011-12', '2013-14', '2015-16', '2017-18']


def load_data(data_path=None, test_size=0.2, random_state=42):
    """
    Load and prepare the NHANES dataset for modelling.

    Returns a dictionary with everything needed for training and evaluation.

    Plain English:
        We split data into training (80%) and testing (20%).
        We standardize features — rescale every column to mean=0, std=1.
        This stops variables with big numbers (BP=120) dominating
        variables with small numbers (BMI=28).
        We then convert everything to PyTorch tensors — the format
        PyTorch needs to do neural network computations.

    What is a tensor?
        A tensor is just a multi-dimensional array — like a numpy array
        but with extra capabilities for automatic differentiation
        (computing gradients). Think of it as a smarter matrix.
    """

    # ── Find data file ─────────────────────────────────────────────────────
    if data_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(here, '..', 'data', 'nhanes_master.csv')

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded: {len(df):,} rows, {len(FEATURE_COLS)} features")

    # ── Extract arrays ─────────────────────────────────────────────────────
    X    = df[FEATURE_COLS].values.astype(np.float32)
    y    = df[OUTCOME_COL].values.astype(np.float32)
    envs = df[ENV_COL].values

    # ── Encode environments as integers ────────────────────────────────────
    # 2011-12 → 0, 2013-14 → 1, 2015-16 → 2, 2017-18 → 3
    env_to_idx = {e: i for i, e in enumerate(ENV_ORDER)}
    env_idx    = np.array([env_to_idx[e] for e in envs])

    # ── Train / test split ─────────────────────────────────────────────────
    # Stratify by both outcome AND environment
    # Ensures fair representation of all cycles in both splits
    strat_key = y.astype(str) + '_' + env_idx.astype(str)

    (X_train, X_test,
     y_train, y_test,
     env_train, env_test) = train_test_split(
        X, y, env_idx,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_key
    )

    # ── Standardize features ───────────────────────────────────────────────
    # Fit scaler ONLY on training data — never on test data
    # Fitting on test data = data leakage = cheating
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # ── Convert to PyTorch tensors ─────────────────────────────────────────
    # torch.FloatTensor = 32-bit float tensor (standard for neural nets)
    # torch.LongTensor  = integer tensor (for environment indices)
    X_train_t   = torch.FloatTensor(X_train)
    X_test_t    = torch.FloatTensor(X_test)
    y_train_t   = torch.FloatTensor(y_train)
    y_test_t    = torch.FloatTensor(y_test)
    env_train_t = torch.LongTensor(env_train)
    env_test_t  = torch.LongTensor(env_test)

    # ── Organize by environment ────────────────────────────────────────────
    # CausTab sees each environment separately during training
    # This is the structural difference from standard ML
    train_envs = {}
    test_envs  = {}

    for e_idx, e_name in enumerate(ENV_ORDER):

        # Training split for this environment
        mask = env_train == e_idx
        train_envs[e_name] = {
            'X': X_train_t[mask],
            'y': y_train_t[mask],
            'n': int(mask.sum())
        }

        # Test split for this environment
        mask = env_test == e_idx
        test_envs[e_name] = {
            'X': X_test_t[mask],
            'y': y_test_t[mask],
            'n': int(mask.sum())
        }

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"{'Split':<12} {'Environment':<12} {'N':>6} {'Hypert.%':>10}")
    print(f"{'─'*50}")
    for e_name, d in train_envs.items():
        prev = d['y'].mean().item() * 100
        print(f"{'Train':<12} {e_name:<12} {d['n']:>6,} {prev:>9.1f}%")
    print(f"{'─'*50}")
    for e_name, d in test_envs.items():
        prev = d['y'].mean().item() * 100
        print(f"{'Test':<12} {e_name:<12} {d['n']:>6,} {prev:>9.1f}%")
    print(f"{'─'*50}")
    print(f"\nTotal train: {len(X_train_t):,} | Total test: {len(X_test_t):,}")
    print(f"Features: {len(FEATURE_COLS)} | Environments: {len(ENV_ORDER)}")

    return {
        # Full arrays
        'X_train':      X_train_t,
        'X_test':       X_test_t,
        'y_train':      y_train_t,
        'y_test':       y_test_t,
        'env_train':    env_train_t,
        'env_test':     env_test_t,
        # Per-environment dicts
        'train_envs':   train_envs,
        'test_envs':    test_envs,
        # Metadata
        'scaler':       scaler,
        'feature_names': FEATURE_COLS,
        'env_order':    ENV_ORDER,
        'n_features':   X_train_t.shape[1],
        'n_envs':       len(ENV_ORDER),
    }


if __name__ == "__main__":
    data = load_data()
    print("\nSample tensor shapes:")
    print(f"  X_train: {data['X_train'].shape}")
    print(f"  y_train: {data['y_train'].shape}")
    print(f"  env_train: {data['env_train'].shape}")
    print("\nData loader working correctly. Ready for modelling.")