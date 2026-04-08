"""
CausTab — UCI Heart Disease Experiment

Dataset: UCI Heart Disease (Janosi et al. 1989)
Source:  UCI ML Repository, ID=45
License: CC BY 4.0

Four hospitals as four environments — institutional shift:
    Cleveland Clinic, USA          → environment 0
    Hungarian Institute, Budapest  → environment 1
    University Hospital, Zurich    → environment 2
    VA Medical Center, Long Beach  → environment 3

Why this matters:
    NHANES showed temporal shift — same population, different years.
    UCI Heart Disease shows institutional shift — same measurements,
    completely different hospitals across three countries.
    This is the real deployment problem: train at one hospital,
    deploy at another. A fundamentally harder and more realistic
    test of distribution shift robustness.

Shift type: institutional
    Different patient demographics per hospital
    Different measurement protocols
    Different disease prevalence
    Different confounding structures

Experimental design:
    Leave-one-hospital-out cross validation
    Train on 3 hospitals, test on the 4th
    Repeat for all 4 combinations
    Report mean AUC across all folds

Plain English:
    We train on Cleveland + Hungary + Switzerland,
    test on VA Long Beach. Then train on Cleveland +
    Hungary + VA, test on Switzerland. And so on.
    Each fold asks: does the model generalize to a
    completely unseen hospital it never trained on?
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
import time
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.impute import SimpleImputer

import urllib.request
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import ERM, IRM, CausTab

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, 'experiments', 'results', 'uci_heart')
PLOTS_DIR   = os.path.join(ROOT, 'experiments', 'plots', 'uci_heart')
DATA_DIR    = os.path.join(ROOT, 'data')

for d in [RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

COLORS = {
    'ERM':     '#4C72B0',
    'IRM':     '#DD8452',
    'CausTab': '#55A868'
}

# Hospital names for display
HOSPITAL_NAMES = {
    0: 'Cleveland\n(USA)',
    1: 'Hungary\n(Budapest)',
    2: 'Switzerland\n(Zurich)',
    3: 'VA Long Beach\n(USA)',
}

CONFIG = {
    'n_epochs':       200,
    'lr':             1e-3,
    'lambda_irm':     1.0,
    'lambda_caustab': 100.0,
    'anneal_epochs':  50,
    'random_state':   42,
    'n_bootstrap':    500,
}

# The 13 standard features used in all published UCI heart disease work
FEATURE_COLS = [
    'age',       # age in years
    'sex',       # 1=male, 0=female
    'cp',        # chest pain type (1-4)
    'trestbps',  # resting blood pressure
    'chol',      # serum cholesterol
    'fbs',       # fasting blood sugar > 120 mg/dl (1=true)
    'restecg',   # resting ECG results (0-2)
    'thalach',   # maximum heart rate achieved
    'exang',     # exercise induced angina (1=yes)
    'oldpeak',   # ST depression induced by exercise
    'slope',     # slope of peak exercise ST segment
    'ca',        # number of major vessels colored by fluoroscopy
    'thal',      # thalassemia type
]


def download_and_prepare():
    """
    Download UCI Heart Disease dataset using urllib.
    Uses the old UCI archive which serves raw files directly.
    No API needed — pure urllib download.
    """
    print("  Downloading UCI Heart Disease dataset...")

    cache_path = os.path.join(DATA_DIR, 'uci_heart_disease.csv')

    if os.path.exists(cache_path):
        print(f"  Loading from cache: {cache_path}")
        df = pd.read_csv(cache_path)
        return df

    # Direct file URLs from UCI ML archive
    # These are the four processed files — 14 features each
    FILES = {
        'cleveland':  'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
        'hungarian':  'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data',
        'switzerland':'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data',
        'va':         'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data',
    }

    # 14 standard column names used in all UCI heart disease work
    COL_NAMES = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
        'ca', 'thal', 'target'
    ]

    HOSPITAL_IDS = {
        'cleveland':   0,
        'hungarian':   1,
        'switzerland': 2,
        'va':          3,
    }

    all_dfs = []

    for name, url in FILES.items():
        print(f"  Downloading {name}...", end=" ")
        try:
            import io
            response = urllib.request.urlopen(url)
            content  = response.read().decode('utf-8')
            # Read as CSV — values separated by commas
            # Missing values marked as '?' — replace with NaN
            sub_df = pd.read_csv(
                io.StringIO(content),
                header    = None,
                names     = COL_NAMES,
                na_values = ['?', '-9', -9.0]
            )
            sub_df['hospital'] = HOSPITAL_IDS[name]
            all_dfs.append(sub_df)
            print(f"done. ({len(sub_df)} rows)")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

    if not all_dfs:
        raise RuntimeError(
            "Could not download any UCI heart disease files. "
            "Check internet connection."
        )

    df = pd.concat(all_dfs, ignore_index=True)

    # Binarize target: 0 = no disease, 1 = disease present
    df['target'] = (df['target'] > 0).astype(int)

    # Print summary
    print(f"\n  Combined dataset: {len(df)} rows")
    for h_id, h_name in HOSPITAL_NAMES.items():
        sub  = df[df['hospital'] == h_id]
        if len(sub) == 0:
            continue
        prev = sub['target'].mean() * 100
        print(f"    {h_name.replace(chr(10),' ')}: "
              f"{len(sub)} patients, {prev:.1f}% heart disease")

    # Save cache
    df.to_csv(cache_path, index=False)
    print(f"  Saved cache: {cache_path}")

    return df
    """
    Download UCI Heart Disease dataset and prepare for experiments.

    The dataset has 4 separate files — one per hospital.
    We download all four and combine with a hospital identifier.

    Plain English:
        ucimlrepo fetches the dataset directly from UCI's API.
        The 'source' column in the original data identifies
        which hospital each row came from — that is our
        environment label.
    """
    print("  Downloading UCI Heart Disease dataset...")

    # Cache path — avoid re-downloading
    cache_path = os.path.join(DATA_DIR, 'uci_heart_disease.csv')

    if os.path.exists(cache_path):
        print(f"  Loading from cache: {cache_path}")
        df = pd.read_csv(cache_path)
        return df

    # Fetch from UCI repository
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # Combine features and target
    df = X.copy()
    df['target'] = (y.values.flatten() > 0).astype(int)
    # target: 0 = no heart disease, 1 = heart disease present
    # Original is 0-4; we binarize to 0 vs 1-4

    # The UCI repository provides a 'source' or location indicator
    # We need to reconstruct hospital labels from known dataset sizes:
    # Cleveland: 303 rows
    # Hungarian: 294 rows
    # Switzerland: 123 rows
    # VA Long Beach: 200 rows
    # Total: 920 rows

    # Assign hospital labels based on known row counts
    hospital_sizes = [303, 294, 123, 200]
    hospital_ids   = []
    for i, size in enumerate(hospital_sizes):
        hospital_ids.extend([i] * size)

    # Handle any size mismatch gracefully
    if len(hospital_ids) != len(df):
        print(f"  Warning: expected {sum(hospital_sizes)} rows, "
              f"got {len(df)}. Adjusting...")
        if len(df) > len(hospital_ids):
            hospital_ids.extend(
                [3] * (len(df) - len(hospital_ids)))
        else:
            hospital_ids = hospital_ids[:len(df)]

    df['hospital'] = hospital_ids

    # Keep only our 13 standard features + target + hospital
    available_features = [f for f in FEATURE_COLS if f in df.columns]
    df = df[available_features + ['target', 'hospital']]

    print(f"  Downloaded: {len(df)} rows, "
          f"{len(available_features)} features")
    print(f"  Hospitals:")
    for h_id, h_name in HOSPITAL_NAMES.items():
        n     = (df['hospital'] == h_id).sum()
        prev  = df.loc[df['hospital']==h_id, 'target'].mean()*100
        print(f"    {h_name.replace(chr(10),' ')}: "
              f"{n} patients, {prev:.1f}% heart disease")

    # Save cache
    df.to_csv(cache_path, index=False)
    print(f"  Saved cache: {cache_path}")

    return df


def prepare_loocv_split(df, test_hospital, feature_cols):
    """
    Prepare one leave-one-hospital-out fold.

    Train: all hospitals except test_hospital
    Test:  test_hospital only

    Plain English:
        We hold out one hospital completely.
        The model trains on the other three and must generalize
        to the held-out hospital it has never seen.
        This is the hardest possible evaluation for institutional shift.
    """
    train_mask = df['hospital'] != test_hospital
    test_mask  = df['hospital'] == test_hospital

    X_train_raw = df.loc[train_mask, feature_cols].values.astype(
        np.float32)
    X_test_raw  = df.loc[test_mask,  feature_cols].values.astype(
        np.float32)
    y_train = df.loc[train_mask, 'target'].values.astype(np.float32)
    y_test  = df.loc[test_mask,  'target'].values.astype(np.float32)
    h_train = df.loc[train_mask, 'hospital'].values

    # Impute missing values — UCI heart disease has many
    # Strategy: median imputation (standard for clinical data)
    # Plain English: replace missing values with the median
    # of that column across training data
    imputer = SimpleImputer(strategy='median')
    X_train_raw = imputer.fit_transform(X_train_raw)
    X_test_raw  = imputer.transform(X_test_raw)

    # Standardize — fit on training only
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test  = scaler.transform(X_test_raw).astype(np.float32)

    # Organize training data by hospital (environment)
    train_hospitals = sorted(
        [h for h in df['hospital'].unique() if h != test_hospital])
    train_envs = {}
    for h in train_hospitals:
        mask = h_train == h
        if mask.sum() == 0:
            continue
        train_envs[HOSPITAL_NAMES[h].replace('\n', ' ')] = {
            'X': torch.FloatTensor(X_train[mask]),
            'y': torch.FloatTensor(y_train[mask]),
            'n': int(mask.sum())
        }

    return {
        'X_train':    torch.FloatTensor(X_train),
        'X_test':     torch.FloatTensor(X_test),
        'y_train':    torch.FloatTensor(y_train),
        'y_test':     torch.FloatTensor(y_test),
        'train_envs': train_envs,
        'n_features': X_train.shape[1],
        'n_train':    len(y_train),
        'n_test':     len(y_test),
        'test_prev':  y_test.mean() * 100,
    }


def compute_ece(probs, labels, n_bins=10):
    bins      = np.linspace(0, 1, n_bins + 1)
    ece       = 0.0
    n_samples = len(labels)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum()/n_samples) * abs(
            probs[mask].mean() - labels[mask].mean()
        )
    return ece


def bootstrap_auc(y_true, probs, n_bootstrap=500, seed=42):
    rng  = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_bootstrap):
        idx  = rng.randint(0, len(y_true), len(y_true))
        y_b, p_b = y_true[idx], probs[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aucs.append(roc_auc_score(y_b, p_b))
    aucs = np.array(aucs)
    return (np.mean(aucs),
            np.percentile(aucs, 2.5),
            np.percentile(aucs, 97.5))


def run_loocv(df, feature_cols, config=CONFIG):
    """
    Run leave-one-hospital-out cross validation.
    Train on 3 hospitals, test on 1. Repeat 4 times.
    """
    hospitals  = sorted(df['hospital'].unique())
    all_results = {
        'ERM': {}, 'IRM': {}, 'CausTab': {}
    }

    print(f"\n  Leave-One-Hospital-Out Cross Validation")
    print(f"  {'='*60}")

    for test_h in hospitals:
        h_name = HOSPITAL_NAMES[test_h].replace('\n', ' ')
        print(f"\n  Fold: Test = {h_name}")

        data = prepare_loocv_split(df, test_h, feature_cols)
        print(f"    Train: {data['n_train']} patients | "
              f"Test: {data['n_test']} patients "
              f"({data['test_prev']:.1f}% positive)")
        print(f"    Train environments: "
              f"{list(data['train_envs'].keys())}")

        # Train all three models
        for ModelClass, name, kwargs in [
            (ERM,     'ERM',     {}),
            (IRM,     'IRM',     {'lambda_irm':    config['lambda_irm']}),
            (CausTab, 'CausTab', {
                'lambda_caustab': config['lambda_caustab'],
                'anneal_epochs':  config['anneal_epochs']}),
        ]:
            torch.manual_seed(config['random_state'])
            model = ModelClass(
                n_features   = data['n_features'],
                lr           = config['lr'],
                random_state = config['random_state'],
                **kwargs
            )
            model.train(
                data['train_envs'],
                n_epochs = config['n_epochs'],
                verbose  = False
            )

            # Evaluate
            probs  = model.predict_proba(data['X_test'])
            y_true = data['y_test'].numpy()
            y_pred = (probs >= 0.5).astype(int)

            auc_mean, auc_lo, auc_hi = bootstrap_auc(
                y_true, probs,
                n_bootstrap = config['n_bootstrap']
            )

            all_results[name][h_name] = {
                'auc':      roc_auc_score(y_true, probs),
                'auc_lo':   auc_lo,
                'auc_hi':   auc_hi,
                'accuracy': accuracy_score(y_true, y_pred),
                'f1':       f1_score(y_true, y_pred,
                                    zero_division=0),
                'ece':      compute_ece(probs, y_true),
                'n_test':   data['n_test'],
            }

        # Print fold results
        print(f"    {'Model':<12} {'AUC':>8} {'95% CI':>20} "
              f"{'ECE':>8}")
        print(f"    {'-'*52}")
        for name in ['ERM', 'IRM', 'CausTab']:
            r = all_results[name][h_name]
            ci = f"[{r['auc_lo']:.3f},{r['auc_hi']:.3f}]"
            print(f"    {name:<12} {r['auc']:>8.4f} "
                  f"{ci:>20} {r['ece']:>8.4f}")

    return all_results


def print_loocv_summary(all_results):
    """Print clean summary table across all folds."""
    print(f"\n{'='*65}")
    print("LOOCV SUMMARY — Mean across all hospital folds")
    print(f"{'='*65}")

    test_hospitals = list(
        list(all_results.values())[0].keys())

    for metric, label in [
        ('auc',      'AUC-ROC'),
        ('accuracy', 'Accuracy'),
        ('f1',       'F1 Score'),
        ('ece',      'ECE (↓)'),
    ]:
        print(f"\n{label}:")
        print(f"  {'Model':<12}", end="")
        for h in test_hospitals:
            short = h.split('\n')[0][:10]
            print(f" {short:>12}", end="")
        print(f" {'Mean':>8} {'Range':>8}")
        print(f"  {'-'*70}")

        for model_name in ['ERM', 'IRM', 'CausTab']:
            vals = [all_results[model_name][h][metric]
                   for h in test_hospitals]
            mean = np.mean(vals)
            rng  = max(vals) - min(vals)
            print(f"  {model_name:<12}", end="")
            for v in vals:
                print(f" {v:>12.4f}", end="")
            marker = " ★" if model_name == 'CausTab' else "  "
            print(f" {mean:>8.4f} {rng:>8.4f}{marker}")


def plot_loocv_results(all_results):
    """
    Main results plot — AUC per hospital fold per model.
    Shows institutional shift robustness.
    """
    test_hospitals = list(
        list(all_results.values())[0].keys())
    n_hospitals    = len(test_hospitals)
    short_names    = [h.replace('\n', '\n') for h in test_hospitals]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "CausTab — UCI Heart Disease\n"
        "Leave-One-Hospital-Out Evaluation "
        "(Institutional Distribution Shift)",
        fontsize=12, fontweight='bold'
    )

    # ── Plot 1: AUC bar chart with CI ──────────────────────────────────────
    ax    = axes[0]
    x     = np.arange(n_hospitals)
    width = 0.25

    for i, model_name in enumerate(['ERM', 'IRM', 'CausTab']):
        aucs   = [all_results[model_name][h]['auc']
                 for h in test_hospitals]
        errors = [
            [all_results[model_name][h]['auc'] -
             all_results[model_name][h]['auc_lo']
             for h in test_hospitals],
            [all_results[model_name][h]['auc_hi'] -
             all_results[model_name][h]['auc']
             for h in test_hospitals]
        ]
        bars = ax.bar(x + i*width, aucs, width,
                     label=model_name,
                     color=COLORS[model_name],
                     alpha=0.85,
                     yerr=errors, capsize=4,
                     error_kw={'linewidth': 1.2})

        for bar, v in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.015,
                   f'{v:.3f}',
                   ha='center', va='bottom', fontsize=7)

    ax.set_xlabel("Test hospital (held out)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC per hospital fold\n(with 95% bootstrap CI)")
    ax.set_xticks(x + width)
    short = [h.split('(')[0].strip() for h in test_hospitals]
    ax.set_xticklabels(short, fontsize=9)
    ax.set_ylim(0.4, 0.95)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Plot 2: Mean AUC summary ───────────────────────────────────────────
    ax2          = axes[1]
    model_names  = ['ERM', 'IRM', 'CausTab']
    mean_aucs    = [np.mean([all_results[m][h]['auc']
                            for h in test_hospitals])
                   for m in model_names]
    mean_ecces   = [np.mean([all_results[m][h]['ece']
                            for h in test_hospitals])
                   for m in model_names]
    colors       = [COLORS[m] for m in model_names]
    x2           = np.arange(len(model_names))

    bars2 = ax2.bar(x2, mean_aucs,
                   color=colors, alpha=0.85)
    bars2[2].set_edgecolor('black')
    bars2[2].set_linewidth(2)

    for bar, auc, ece in zip(bars2, mean_aucs, mean_ecces):
        ax2.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'AUC={auc:.3f}\nECE={ece:.3f}',
                ha='center', va='bottom', fontsize=9)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(model_names, fontsize=11)
    ax2.set_ylabel("Mean AUC-ROC across hospital folds")
    ax2.set_title("Mean performance across all folds\n"
                 "(lower ECE = better calibrated)")
    ax2.set_ylim(0.4, 0.95)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'uci_loocv_results.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: uci_loocv_results.png")


def plot_hospital_shift_evidence(df, feature_cols):
    """
    Show distribution shift evidence across hospitals.
    Analogous to Table 3 from NHANES exploration —
    shows which features shift across hospitals.
    This motivates why institutional shift is a real problem.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "UCI Heart Disease — Institutional Distribution Shift Evidence\n"
        "Feature distributions differ substantially across hospitals",
        fontsize=11, fontweight='bold'
    )

    # ── Plot 1: Disease prevalence per hospital ────────────────────────────
    ax    = axes[0]
    prevs = []
    names = []
    ns    = []
    for h_id in sorted(df['hospital'].unique()):
        sub   = df[df['hospital'] == h_id]
        prev  = sub['target'].mean() * 100
        name  = HOSPITAL_NAMES[h_id].replace('\n', ' ')
        prevs.append(prev)
        names.append(name)
        ns.append(len(sub))

    colors_h = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    bars     = ax.bar(names, prevs, color=colors_h, alpha=0.85)
    for bar, p, n in zip(bars, prevs, ns):
        ax.text(bar.get_x() + bar.get_width()/2,
               bar.get_height() + 0.5,
               f'{p:.1f}%\n(n={n})',
               ha='center', va='bottom', fontsize=9)

    ax.set_ylabel("Heart disease prevalence (%)")
    ax.set_title("Disease prevalence per hospital\n"
                "(substantial variation = institutional shift)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Plot 2: Correlation shift across hospitals ─────────────────────────
    ax2          = axes[1]
    top_features = ['age', 'thalach', 'oldpeak', 'ca', 'thal']
    available    = [f for f in top_features if f in df.columns]

    corr_data = {}
    for h_id in sorted(df['hospital'].unique()):
        sub = df[df['hospital'] == h_id]
        corr_data[HOSPITAL_NAMES[h_id].replace('\n', ' ')] = [
            sub[f].corr(sub['target'])
            for f in available
        ]

    corr_df = pd.DataFrame(
        corr_data, index=available)

    import seaborn as sns
    sns.heatmap(corr_df, annot=True, fmt=".2f",
               cmap="RdBu_r", center=0,
               ax=ax2, cbar_kws={'shrink': 0.8})
    ax2.set_title(
        "Feature-outcome correlations per hospital\n"
        "(variation across columns = institutional shift)")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'uci_shift_evidence.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: uci_shift_evidence.png")


def compute_uci_sdi(df, feature_cols):
    """
    Compute SDI for UCI Heart Disease dataset.
    Allows direct comparison with NHANES SDI.

    We expect higher SDI than NHANES because institutional
    shift is more severe than temporal shift.
    """
    hospitals = sorted(df['hospital'].unique())

    # Compute correlation of each feature with target per hospital
    all_corrs = {f: [] for f in feature_cols
                if f in df.columns}

    for h_id in hospitals:
        sub = df[df['hospital'] == h_id].dropna(
            subset=feature_cols + ['target'])
        for f in all_corrs:
            if sub[f].std() > 0:
                c = sub[f].corr(sub['target'])
                all_corrs[f].append(c)

    # Compute range of correlations per feature
    feature_ranges = {}
    feature_means  = {}
    for f, corrs in all_corrs.items():
        if len(corrs) >= 2:
            feature_ranges[f] = max(corrs) - min(corrs)
            feature_means[f]  = abs(np.mean(corrs))

    if not feature_ranges:
        return {'sdi': 0.0}

    # Classify features as stable vs shifting
    median_range = np.median(list(feature_ranges.values()))
    spurious_corr = np.mean([
        feature_means[f]
        for f, r in feature_ranges.items()
        if r > median_range
    ])
    causal_corr = np.mean([
        feature_means[f]
        for f, r in feature_ranges.items()
        if r <= median_range
    ])
    mean_shift = np.mean(list(feature_ranges.values()))

    sdi = (spurious_corr * mean_shift) / (
        causal_corr * (1 - mean_shift) + 1e-8
    ) if causal_corr > 0 else 0.0

    print(f"\n  UCI Heart Disease SDI = {sdi:.4f}")
    print(f"  Mean feature correlation range: {mean_shift:.4f}")
    print(f"  (Compare: NHANES SDI ≈ low, "
          f"institutional shift expected higher)")

    return {
        'sdi':            round(sdi, 4),
        'mean_shift':     round(mean_shift, 4),
        'spurious_corr':  round(spurious_corr, 4),
        'causal_corr':    round(causal_corr, 4),
        'feature_ranges': feature_ranges,
    }


def save_uci_results(all_results, sdi_info, df):
    """Save all UCI results as CSV and formatted TXT."""
    test_hospitals = list(list(all_results.values())[0].keys())

    # Detailed results CSV
    rows = []
    for model_name in ['ERM', 'IRM', 'CausTab']:
        for h_name in test_hospitals:
            r = all_results[model_name][h_name]
            rows.append({
                'Model':         model_name,
                'Test_Hospital': h_name,
                'N_test':        r['n_test'],
                'AUC':           round(r['auc'],      4),
                'AUC_CI_lo':     round(r['auc_lo'],   4),
                'AUC_CI_hi':     round(r['auc_hi'],   4),
                'Accuracy':      round(r['accuracy'],  4),
                'F1':            round(r['f1'],        4),
                'ECE':           round(r['ece'],       4),
            })

    df_results = pd.DataFrame(rows)
    df_results.to_csv(
        os.path.join(RESULTS_DIR, 'uci_loocv_results.csv'),
        index=False
    )

    # Summary CSV
    summary_rows = []
    for model_name in ['ERM', 'IRM', 'CausTab']:
        aucs = [all_results[model_name][h]['auc']
               for h in test_hospitals]
        eces = [all_results[model_name][h]['ece']
               for h in test_hospitals]
        summary_rows.append({
            'Model':    model_name,
            'Mean_AUC': round(np.mean(aucs), 4),
            'Std_AUC':  round(np.std(aucs),  4),
            'Mean_ECE': round(np.mean(eces),  4),
            'SDI':      sdi_info['sdi'],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(RESULTS_DIR, 'uci_summary.csv'),
        index=False
    )

    # Paper-ready TXT
    txt_path = os.path.join(RESULTS_DIR, 'uci_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("UCI Heart Disease — Leave-One-Hospital-Out Results\n")
        f.write("Institutional distribution shift evaluation\n")
        f.write(f"SDI = {sdi_info['sdi']:.4f}\n")
        f.write("="*72 + "\n")
        f.write("AUC-ROC with 95% Bootstrap CI\n")
        f.write(f"{'Model':<12}")
        for h in test_hospitals:
            short = h.split('(')[0].strip()[:10]
            f.write(f" {short:>20}")
        f.write(f" {'Mean':>8} {'Range':>8}\n")
        f.write("-"*72 + "\n")
        for model_name in ['ERM', 'IRM', 'CausTab']:
            aucs = [all_results[model_name][h]['auc']
                   for h in test_hospitals]
            mean = np.mean(aucs)
            rng  = max(aucs) - min(aucs)
            f.write(f"{model_name:<12}")
            for h in test_hospitals:
                r  = all_results[model_name][h]
                ci = (f"{r['auc']:.3f}"
                      f"[{r['auc_lo']:.3f},"
                      f"{r['auc_hi']:.3f}]")
                f.write(f" {ci:>20}")
            marker = " ★" if model_name == 'CausTab' else "  "
            f.write(f" {mean:>8.4f} {rng:>8.4f}{marker}\n")
        f.write("-"*72 + "\n")
        f.write("★ = proposed method\n")

    print(f"  Saved: uci_loocv_results.csv")
    print(f"  Saved: uci_summary.csv")
    print(f"  Saved: uci_results.txt")

    return df_results, summary_df


if __name__ == "__main__":
    print("="*65)
    print("CausTab — UCI HEART DISEASE EXPERIMENT")
    print("="*65)
    print("Shift type: Institutional")
    print("Design:     Leave-one-hospital-out CV")
    print("Hospitals:  Cleveland, Hungary, Switzerland, VA Long Beach")
    print("="*65)

    # ── Download and prepare data ──────────────────────────────────────────
    print("\n[1/5] Loading data...")
    df = download_and_prepare()

    # Use available features
    feature_cols = [f for f in FEATURE_COLS if f in df.columns]
    print(f"  Features available: {feature_cols}")
    print(f"  Total patients: {len(df)}")

    # ── Compute SDI ────────────────────────────────────────────────────────
    print("\n[2/5] Computing Spurious Dominance Index...")
    sdi_info = compute_uci_sdi(df, feature_cols)

    # ── Plot shift evidence ────────────────────────────────────────────────
    print("\n[3/5] Plotting shift evidence...")
    plot_hospital_shift_evidence(df, feature_cols)

    # ── Run LOOCV ──────────────────────────────────────────────────────────
    print("\n[4/5] Running leave-one-hospital-out CV...")
    start       = time.time()
    all_results = run_loocv(df, feature_cols)
    elapsed     = time.time() - start
    print(f"\n  Total time: {elapsed:.1f}s")

    # ── Print summary ──────────────────────────────────────────────────────
    print_loocv_summary(all_results)

    # ── Generate plots ─────────────────────────────────────────────────────
    print("\n[5/5] Generating plots and saving results...")
    plot_loocv_results(all_results)

    # ── Save results ───────────────────────────────────────────────────────
    df_results, summary_df = save_uci_results(
        all_results, sdi_info, df)

    print(f"\n{'='*65}")
    print("UCI HEART DISEASE EXPERIMENT COMPLETE")
    print(f"Plots:   experiments/plots/uci_heart/")
    print(f"Results: experiments/results/uci_heart/")
    print(f"{'='*65}")
    print(f"\n{summary_df.to_string(index=False)}")