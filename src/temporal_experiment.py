"""
CausTab — Temporal Forward-Chaining Experiment

Experimental design:
    Split B: Train [2011-12, 2013-14] → Test [2015-16, 2017-18]
    Split C: Train [2011-12, 2013-14, 2015-16] → Test [2017-18]

Why this is the right design:
    Models are trained on past environments only.
    They are tested on future environments they never saw.
    This mimics real deployment — you train today, deploy tomorrow.
    Distribution shift is now genuine, not artificial.

Plain English:
    Imagine training a model on 2011-2014 patient data
    then deploying it in 2017. Does it still work?
    That is exactly what we test here.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve
from sklearn.utils import resample as sklearn_resample
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import ERM, IRM, CausTab

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(ROOT, 'data', 'nhanes_master.csv')
RESULTS_DIR = os.path.join(ROOT, 'experiments', 'results', 'temporal_split')
PLOTS_DIR   = os.path.join(ROOT, 'experiments', 'plots', 'temporal_split')
MODELS_DIR  = os.path.join(ROOT, 'experiments', 'saved_models', 'temporal_split')

for d in [RESULTS_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'INDFMPIR', 'DMDEDUC2',
    'BPXSY1', 'BPXDI1', 'BPXSY2', 'BPXDI2', 'BMXBMI', 'BMXWAIST',
]
OUTCOME_COL = 'hypertension'
ENV_COL     = 'environment'
ENV_ORDER   = ['2011-12', '2013-14', '2015-16', '2017-18']

COLORS = {
    'ERM':     '#4C72B0',
    'IRM':     '#DD8452',
    'CausTab': '#55A868'
}

# ── Temporal split definitions ─────────────────────────────────────────────────
# Plain English:
#   Split B = train on 2 early cycles, test on 2 later cycles
#   Split C = train on 3 cycles, test on the most recent cycle only
#   Both always train on past, test on future — never the reverse

TEMPORAL_SPLITS = {
    'Split_B': {
        'train': ['2011-12', '2013-14'],
        'test':  ['2015-16', '2017-18'],
        'description': 'Train on 2011-2014, Test on 2015-2018'
    },
    'Split_C': {
        'train': ['2011-12', '2013-14', '2015-16'],
        'test':  ['2017-18'],
        'description': 'Train on 2011-2016, Test on 2017-2018'
    },
}

CONFIG = {
    'n_epochs':       200,
    'lr':             1e-3,
    'lambda_irm':     1.0,
    'lambda_caustab': 1.0,
    'anneal_epochs':  50,
    'random_state':   42,
}


def prepare_temporal_split(df, train_envs_list, test_envs_list):
    """
    Prepare data for one temporal split.

    Plain English:
        We take rows belonging to train cycles → training set
        We take rows belonging to test cycles  → test set
        Scaler is fit on training data only — never on test data
        Each split returns data organized by environment
        so CausTab can apply its per-environment penalty
    """
    train_mask = df[ENV_COL].isin(train_envs_list)
    test_mask  = df[ENV_COL].isin(test_envs_list)

    X_train_raw = df.loc[train_mask, FEATURE_COLS].values.astype(np.float32)
    X_test_raw  = df.loc[test_mask,  FEATURE_COLS].values.astype(np.float32)
    y_train     = df.loc[train_mask, OUTCOME_COL].values.astype(np.float32)
    y_test      = df.loc[test_mask,  OUTCOME_COL].values.astype(np.float32)
    env_train   = df.loc[train_mask, ENV_COL].values
    env_test    = df.loc[test_mask,  ENV_COL].values

    # Standardize — fit on train only
    scaler      = StandardScaler()
    X_train     = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test      = scaler.transform(X_test_raw).astype(np.float32)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    X_test_t  = torch.FloatTensor(X_test)
    y_train_t = torch.FloatTensor(y_train)
    y_test_t  = torch.FloatTensor(y_test)

    # Organize by environment
    train_envs_dict = {}
    for e in train_envs_list:
        mask = env_train == e
        train_envs_dict[e] = {
            'X': X_train_t[mask],
            'y': y_train_t[mask],
            'n': int(mask.sum())
        }

    test_envs_dict = {}
    for e in test_envs_list:
        mask = env_test == e
        test_envs_dict[e] = {
            'X': X_test_t[mask],
            'y': y_test_t[mask],
            'n': int(mask.sum())
        }

    return {
        'X_train':      X_train_t,
        'X_test':       X_test_t,
        'y_train':      y_train_t,
        'y_test':       y_test_t,
        'train_envs':   train_envs_dict,
        'test_envs':    test_envs_dict,
        'n_features':   X_train_t.shape[1],
        'train_env_list': train_envs_list,
        'test_env_list':  test_envs_list,
    }


def compute_ece(probs, labels, n_bins=10):
    bins      = np.linspace(0, 1, n_bins + 1)
    ece       = 0.0
    n_samples = len(labels)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n_samples) * abs(
            probs[mask].mean() - labels[mask].mean()
        )
    return ece


def bootstrap_auc(y_true, probs, n_bootstrap=1000, seed=42):
    rng  = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y_true), len(y_true))
        y_b, p_b = y_true[idx], probs[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aucs.append(roc_auc_score(y_b, p_b))
    aucs = np.array(aucs)
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def train_models(data, split_name, config=CONFIG):
    """Train all three models on a temporal split."""
    print(f"\n  Training ERM...")
    start = time.time()
    erm   = ERM(n_features=data['n_features'],
               lr=config['lr'],
               random_state=config['random_state'])
    erm.train(data['train_envs'], n_epochs=config['n_epochs'], verbose=False)
    print(f"    Done in {time.time()-start:.1f}s | "
          f"Final loss: {erm.train_losses[-1]:.4f}")

    print(f"  Training IRM...")
    start = time.time()
    irm   = IRM(n_features=data['n_features'],
               lr=config['lr'],
               lambda_irm=config['lambda_irm'],
               random_state=config['random_state'])
    irm.train(data['train_envs'], n_epochs=config['n_epochs'], verbose=False)
    print(f"    Done in {time.time()-start:.1f}s | "
          f"Final loss: {irm.train_losses[-1]:.4f}")

    print(f"  Training CausTab...")
    start   = time.time()
    caustab = CausTab(n_features=data['n_features'],
                     lr=config['lr'],
                     lambda_caustab=config['lambda_caustab'],
                     anneal_epochs=config['anneal_epochs'],
                     random_state=config['random_state'])
    caustab.train(data['train_envs'],
                 n_epochs=config['n_epochs'], verbose=False)
    print(f"    Done in {time.time()-start:.1f}s | "
          f"Final loss: {caustab.train_losses[-1]:.4f}")

    # Save models
    for name, model in [('erm', erm), ('irm', irm), ('caustab', caustab)]:
        path = os.path.join(MODELS_DIR,
                           f"{split_name.lower()}_{name}_model.pt")
        torch.save(model.model.state_dict(), path)

    return {'ERM': erm, 'IRM': irm, 'CausTab': caustab}


def evaluate_models(models_dict, test_envs_dict, test_env_list):
    """Evaluate all models on all test environments."""
    all_results = {}

    for model_name, model in models_dict.items():
        all_results[model_name] = {}

        for env_name in test_env_list:
            env_data = test_envs_dict[env_name]
            probs    = model.predict_proba(env_data['X'])
            y_true   = env_data['y'].numpy()
            y_pred   = (probs >= 0.5).astype(int)

            auc_mean, auc_lo, auc_hi = bootstrap_auc(y_true, probs)

            all_results[model_name][env_name] = {
                'auc':      roc_auc_score(y_true, probs),
                'auc_lo':   auc_lo,
                'auc_hi':   auc_hi,
                'accuracy': accuracy_score(y_true, y_pred),
                'f1':       f1_score(y_true, y_pred, zero_division=0),
                'ece':      compute_ece(probs, y_true),
                'probs':    probs,
                'y_true':   y_true,
                'n':        len(y_true)
            }

    return all_results


def print_results(all_results, test_env_list, split_name, description):
    """Print clean results table to terminal."""
    print(f"\n{'='*70}")
    print(f"RESULTS — {split_name}: {description}")
    print(f"{'='*70}")

    for metric, label in [('auc', 'AUC-ROC'),
                           ('accuracy', 'Accuracy'),
                           ('f1', 'F1 Score'),
                           ('ece', 'ECE (↓)')]:
        print(f"\n{label}:")
        print(f"  {'Model':<12}", end="")
        for env in test_env_list:
            print(f" {env:>12}", end="")
        print(f"  {'Range':>8}  {'Mean':>8}")
        print(f"  {'-'*60}")

        for model_name, env_results in all_results.items():
            vals = [env_results[e][metric] for e in test_env_list]
            rng  = max(vals) - min(vals)
            mean = np.mean(vals)
            print(f"  {model_name:<12}", end="")
            for v in vals:
                print(f" {v:>12.4f}", end="")
            print(f"  {rng:>8.4f}  {mean:>8.4f}")

    # Bootstrap CIs for AUC
    print(f"\nAUC 95% Bootstrap CIs:")
    print(f"  {'Model':<12}", end="")
    for env in test_env_list:
        print(f" {env:>22}", end="")
    print()
    print(f"  {'-'*70}")
    for model_name, env_results in all_results.items():
        print(f"  {model_name:<12}", end="")
        for env in test_env_list:
            r = env_results[env]
            ci = f"{r['auc']:.3f} [{r['auc_lo']:.3f},{r['auc_hi']:.3f}]"
            print(f" {ci:>22}", end="")
        print()


def save_results(all_results, test_env_list,
                split_name, description):
    """Save results as CSV and formatted TXT."""
    rows = []
    for model_name, env_results in all_results.items():
        for env_name in test_env_list:
            r = env_results[env_name]
            rows.append({
                'Split':       split_name,
                'Model':       model_name,
                'Environment': env_name,
                'N':           r['n'],
                'AUC':         round(r['auc'],      4),
                'AUC_CI_lo':   round(r['auc_lo'],   4),
                'AUC_CI_hi':   round(r['auc_hi'],   4),
                'Accuracy':    round(r['accuracy'],  4),
                'F1':          round(r['f1'],        4),
                'ECE':         round(r['ece'],       4),
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(
        RESULTS_DIR, f'{split_name.lower()}_results.csv')
    df.to_csv(csv_path, index=False)

    # Paper-ready TXT
    txt_path = os.path.join(
        RESULTS_DIR, f'{split_name.lower()}_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"CausTab — {split_name}: {description}\n")
        f.write(f"Metric: AUC-ROC with 95% Bootstrap CI\n")
        f.write("="*72 + "\n")
        f.write(f"{'Model':<12}")
        for env in test_env_list:
            f.write(f" {env:>22}")
        f.write(f"  {'Range':>8}  {'Mean':>8}\n")
        f.write("-"*72 + "\n")
        for model_name, env_results in all_results.items():
            aucs = [env_results[e]['auc'] for e in test_env_list]
            rng  = max(aucs) - min(aucs)
            mean = np.mean(aucs)
            f.write(f"{model_name:<12}")
            for env in test_env_list:
                r  = env_results[env]
                ci = f"{r['auc']:.3f}[{r['auc_lo']:.3f},{r['auc_hi']:.3f}]"
                f.write(f" {ci:>22}")
            marker = " ★" if model_name == 'CausTab' else "  "
            f.write(f"  {rng:>8.4f}{marker}  {mean:>8.4f}\n")
        f.write("-"*72 + "\n")
        f.write("★ = proposed method\n")

    print(f"\n  Saved: {split_name.lower()}_results.csv / .txt")
    return df


def plot_results(all_results, test_env_list,
                split_name, description):
    """Generate all plots for one temporal split."""

    # ── Plot 1: AUC comparison ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"CausTab — {split_name}\n{description}",
        fontsize=12, fontweight='bold'
    )

    # Bar chart with CI error bars
    ax    = axes[0]
    x     = np.arange(len(test_env_list))
    width = 0.25

    for i, (model_name, env_results) in enumerate(all_results.items()):
        aucs   = [env_results[e]['auc']    for e in test_env_list]
        errors = [
            [env_results[e]['auc'] - env_results[e]['auc_lo']
             for e in test_env_list],
            [env_results[e]['auc_hi'] - env_results[e]['auc']
             for e in test_env_list]
        ]
        bars = ax.bar(x + i*width, aucs, width,
                     label=model_name,
                     color=COLORS[model_name],
                     alpha=0.85,
                     yerr=errors, capsize=4,
                     error_kw={'linewidth': 1.2})
        for bar, v in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.003,
                   f'{v:.3f}', ha='center',
                   va='bottom', fontsize=7)

    ax.set_xlabel("Test environment (unseen during training)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC-ROC per test environment\n(with 95% bootstrap CI)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(test_env_list)
    ax.set_ylim(0.70, 0.90)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Line plot — degradation trend
    ax2 = axes[1]
    for model_name, env_results in all_results.items():
        aucs = [env_results[e]['auc'] for e in test_env_list]
        ax2.plot(test_env_list, aucs,
                marker='o', linewidth=2, markersize=8,
                label=model_name,
                color=COLORS[model_name])
        # CI band
        lowers = [env_results[e]['auc_lo'] for e in test_env_list]
        uppers = [env_results[e]['auc_hi'] for e in test_env_list]
        ax2.fill_between(test_env_list, lowers, uppers,
                        alpha=0.12, color=COLORS[model_name])

    ax2.set_xlabel("Test environment")
    ax2.set_ylabel("AUC-ROC")
    ax2.set_title("AUC trend with confidence bands\n"
                 "(flatter = more robust)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR,
                       f'{split_name.lower()}_auc_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {split_name.lower()}_auc_comparison.png")

    # ── Plot 2: All metrics side by side ───────────────────────────────────
    metrics     = ['auc', 'accuracy', 'f1', 'ece']
    metric_labs = ['AUC-ROC', 'Accuracy', 'F1 Score', 'ECE (↓ better)']

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"CausTab — All Metrics — {split_name}\n{description}",
        fontsize=12, fontweight='bold'
    )

    for ax, metric, label in zip(axes.flat, metrics, metric_labs):
        x     = np.arange(len(test_env_list))
        width = 0.25
        for i, (model_name, env_results) in enumerate(all_results.items()):
            vals = [env_results[e][metric] for e in test_env_list]
            ax.bar(x + i*width, vals, width,
                  label=model_name,
                  color=COLORS[model_name],
                  alpha=0.85)
        ax.set_title(label)
        ax.set_xticks(x + width)
        ax.set_xticklabels(test_env_list, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR,
                       f'{split_name.lower()}_all_metrics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {split_name.lower()}_all_metrics.png")


def plot_combined_summary(all_split_results):
    """
    Final summary plot comparing both splits side by side.
    This is the paper's headline figure for the temporal experiment.
    Shows CausTab's stability advantage consistently across both splits.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "CausTab — Temporal Forward-Chaining Evaluation\n"
        "Mean AUC across test environments (with range bars)",
        fontsize=12, fontweight='bold'
    )

    for ax, (split_name, split_data) in zip(
            axes, all_split_results.items()):

        all_results   = split_data['results']
        test_env_list = split_data['test_envs']
        description   = split_data['description']

        model_names = list(all_results.keys())
        means = []
        ranges = []
        cis_lo = []
        cis_hi = []

        for model_name in model_names:
            aucs  = [all_results[model_name][e]['auc']
                    for e in test_env_list]
            lo    = [all_results[model_name][e]['auc_lo']
                    for e in test_env_list]
            hi    = [all_results[model_name][e]['auc_hi']
                    for e in test_env_list]
            means.append(np.mean(aucs))
            ranges.append(max(aucs) - min(aucs))
            cis_lo.append(np.mean(lo))
            cis_hi.append(np.mean(hi))

        x      = np.arange(len(model_names))
        colors = [COLORS[m] for m in model_names]
        errors = [
            [m - l for m, l in zip(means, cis_lo)],
            [h - m for m, h in zip(means, cis_hi)]
        ]

        bars = ax.bar(x, means, color=colors, alpha=0.85,
                     yerr=errors, capsize=6,
                     error_kw={'linewidth': 1.5})

        # Annotate range on each bar
        for bar, m, r in zip(bars, means, ranges):
            ax.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.004,
                   f'AUC={m:.3f}\nRange={r:.4f}',
                   ha='center', va='bottom', fontsize=8)

        ax.set_title(f"{split_name}\n{description}", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=10)
        ax.set_ylabel("Mean AUC-ROC")
        ax.set_ylim(0.72, 0.88)
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'combined_temporal_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved plot: combined_temporal_summary.png  ← HEADLINE FIGURE")


if __name__ == "__main__":

    # ── Load raw data ──────────────────────────────────────────────────────
    print("="*60)
    print("CausTab — TEMPORAL FORWARD-CHAINING EXPERIMENT")
    print("="*60)
    print("\nLoading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded: {len(df):,} rows")

    all_split_results = {}

    # ── Run each temporal split ────────────────────────────────────────────
    for split_name, split_cfg in TEMPORAL_SPLITS.items():
        print(f"\n{'='*60}")
        print(f"{split_name}: {split_cfg['description']}")
        print(f"  Train environments: {split_cfg['train']}")
        print(f"  Test environments:  {split_cfg['test']}")
        print(f"{'='*60}")

        # Prepare data
        data = prepare_temporal_split(
            df,
            split_cfg['train'],
            split_cfg['test']
        )

        # Print split summary
        print(f"\n  Training data:")
        for e, d in data['train_envs'].items():
            prev = d['y'].mean().item() * 100
            print(f"    {e}: {d['n']:,} samples, {prev:.1f}% hypertension")
        print(f"  Test data:")
        for e, d in data['test_envs'].items():
            prev = d['y'].mean().item() * 100
            print(f"    {e}: {d['n']:,} samples, {prev:.1f}% hypertension")

        # Train
        print(f"\n  Training models...")
        models_dict = train_models(data, split_name)

        # Evaluate
        print(f"\n  Evaluating...")
        results = evaluate_models(
            models_dict,
            data['test_envs'],
            split_cfg['test']
        )

        # Print results
        print_results(results, split_cfg['test'],
                     split_name, split_cfg['description'])

        # Save results and plots
        print(f"\n  Saving...")
        save_results(results, split_cfg['test'],
                    split_name, split_cfg['description'])
        plot_results(results, split_cfg['test'],
                    split_name, split_cfg['description'])

        all_split_results[split_name] = {
            'results':     results,
            'test_envs':   split_cfg['test'],
            'description': split_cfg['description']
        }

    # ── Combined summary plot ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Generating combined summary...")
    plot_combined_summary(all_split_results)

    # ── Final combined CSV ─────────────────────────────────────────────────
    all_rows = []
    for split_name, split_data in all_split_results.items():
        for model_name, env_results in split_data['results'].items():
            for env_name, r in env_results.items():
                all_rows.append({
                    'Split':       split_name,
                    'Model':       model_name,
                    'Environment': env_name,
                    'AUC':         round(r['auc'],      4),
                    'AUC_CI_lo':   round(r['auc_lo'],   4),
                    'AUC_CI_hi':   round(r['auc_hi'],   4),
                    'Accuracy':    round(r['accuracy'],  4),
                    'F1':          round(r['f1'],        4),
                    'ECE':         round(r['ece'],       4),
                })

    combined_df = pd.DataFrame(all_rows)
    combined_df.to_csv(
        os.path.join(RESULTS_DIR, 'all_temporal_results.csv'), index=False
    )
    print(f"  Saved: all_temporal_results.csv")

    print(f"\n{'='*60}")
    print("TEMPORAL EXPERIMENT COMPLETE")
    print(f"Plots:   experiments/plots/temporal_split/")
    print(f"Results: experiments/results/temporal_split/")
    print(f"Models:  experiments/saved_models/temporal_split/")
    print(f"{'='*60}")