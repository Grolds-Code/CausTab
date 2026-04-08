"""
CausTab — Statistical Significance & Sensitivity Analysis

What this file does:
1. Bootstrap confidence intervals for AUC per model per environment
   Plain English: we resample the test data 1000 times and recompute
   AUC each time. The spread of those 1000 AUCs tells us how
   uncertain our estimate is. This is called a confidence interval.

2. Sensitivity analysis for lambda_caustab
   We retrain CausTab with different penalty strengths and show
   performance stays stable — the method is not fragile to this choice.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data, ENV_ORDER
from models import ERM, IRM, CausTab
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, 'experiments', 'results')
PLOTS_DIR   = os.path.join(ROOT, 'experiments', 'plots')
MODELS_DIR  = os.path.join(ROOT, 'experiments', 'saved_models')

for d in [RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

COLORS = {
    'ERM':     '#4C72B0',
    'IRM':     '#DD8452',
    'CausTab': '#55A868'
}


def bootstrap_auc(y_true, probs, n_bootstrap=1000, ci=0.95, seed=42):
    """
    Bootstrap confidence interval for AUC.

    Plain English:
        We repeatedly resample our test data with replacement
        (some samples appear twice, some not at all) and compute
        AUC each time. After 1000 repetitions we have a distribution
        of AUC values. The middle 95% of that distribution is our
        95% confidence interval.

        If two models' confidence intervals don't overlap,
        the difference is statistically significant.
    """
    rng  = np.random.RandomState(seed)
    aucs = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y_true), len(y_true))
        y_b = y_true[idx]
        p_b = probs[idx]
        # Skip if only one class present in resample
        if len(np.unique(y_b)) < 2:
            continue
        aucs.append(roc_auc_score(y_b, p_b))

    aucs = np.array(aucs)
    alpha = (1 - ci) / 2
    lower = np.percentile(aucs, alpha * 100)
    upper = np.percentile(aucs, (1 - alpha) * 100)
    mean  = np.mean(aucs)

    return mean, lower, upper


def run_bootstrap_analysis(models_dict, test_envs, n_bootstrap=1000):
    """
    Run bootstrap CI for all models across all environments.
    """
    print("\n" + "="*70)
    print("BOOTSTRAP CONFIDENCE INTERVALS (95%, n=1000)")
    print("="*70)
    print("Format: AUC [lower, upper]")

    all_ci = {}

    for model_name, model in models_dict.items():
        all_ci[model_name] = {}
        print(f"\n{model_name}:")

        for env_name in ENV_ORDER:
            env_data = test_envs[env_name]
            probs    = model.predict_proba(env_data['X'])
            y_true   = env_data['y'].numpy()

            mean, lower, upper = bootstrap_auc(
                y_true, probs, n_bootstrap=n_bootstrap
            )
            all_ci[model_name][env_name] = {
                'mean': mean, 'lower': lower, 'upper': upper
            }
            print(f"  {env_name}: {mean:.4f} [{lower:.4f}, {upper:.4f}]")

    return all_ci


def plot_bootstrap_ci(all_ci):
    """
    Plot AUC with confidence intervals per environment.
    Overlapping CIs = not statistically distinguishable.
    Non-overlapping = significant difference.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    fig.suptitle(
        "AUC-ROC with 95% Bootstrap Confidence Intervals\n"
        "per NHANES Survey Cycle",
        fontsize=12, fontweight='bold'
    )

    model_names = list(all_ci.keys())
    x           = np.arange(len(model_names))

    for ax, env_name in zip(axes, ENV_ORDER):
        means  = [all_ci[m][env_name]['mean']  for m in model_names]
        lowers = [all_ci[m][env_name]['lower'] for m in model_names]
        uppers = [all_ci[m][env_name]['upper'] for m in model_names]
        errors = [
            [m - l for m, l in zip(means, lowers)],
            [u - m for m, u in zip(means, uppers)]
        ]

        colors = [COLORS[m] for m in model_names]
        bars   = ax.bar(x, means, color=colors, alpha=0.8,
                       yerr=errors, capsize=5,
                       error_kw={'linewidth': 1.5})

        ax.set_title(env_name, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=8, rotation=15)
        ax.set_ylim(0.75, 0.87)
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if ax == axes[0]:
            ax.set_ylabel("AUC-ROC")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'bootstrap_ci.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved plot: bootstrap_ci.png")


def save_ci_table(all_ci):
    """Save CI results as paper-ready table."""
    rows = []
    for model_name in all_ci:
        for env_name in ENV_ORDER:
            ci = all_ci[model_name][env_name]
            rows.append({
                'Model':       model_name,
                'Environment': env_name,
                'AUC':         round(ci['mean'],  4),
                'CI_lower':    round(ci['lower'], 4),
                'CI_upper':    round(ci['upper'], 4),
                'CI_width':    round(ci['upper'] - ci['lower'], 4),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'bootstrap_ci.csv'), index=False)

    # Paper-ready TXT
    txt_path = os.path.join(RESULTS_DIR, 'bootstrap_ci.txt')
    with open(txt_path, 'w') as f:
        f.write("Bootstrap Confidence Intervals (95%, n=1000 resamples)\n")
        f.write("Format: AUC [95% CI lower, upper]\n")
        f.write("="*72 + "\n")
        f.write(f"{'Model':<12}")
        for env in ENV_ORDER:
            f.write(f" {env:>18}")
        f.write("\n" + "-"*72 + "\n")
        for model_name in all_ci:
            f.write(f"{model_name:<12}")
            for env in ENV_ORDER:
                ci = all_ci[model_name][env]
                cell = f"{ci['mean']:.3f} [{ci['lower']:.3f},{ci['upper']:.3f}]"
                f.write(f" {cell:>18}")
            f.write("\n")

    print(f"  Saved: bootstrap_ci.csv / .txt")
    return df


def sensitivity_analysis(data, lambdas=None):
    """
    Retrain CausTab with different lambda values.
    Shows results are robust to this hyperparameter choice.

    Plain English:
        lambda controls how hard we penalize spurious features.
        If results only work for lambda=1.0, reviewers will be skeptical.
        We show it works across a range — that's a robust method.
    """
    if lambdas is None:
        lambdas = [0.1, 0.5, 1.0, 2.0, 5.0]

    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS — lambda_caustab")
    print("="*60)
    print("Retraining CausTab with different penalty strengths...")
    print(f"{'Lambda':<10} {'Mean AUC':>10} {'AUC Range':>12} "
          f"{'2011-12':>10} {'2017-18':>10}")
    print("-"*55)

    sensitivity_rows = []

    for lam in lambdas:
        torch.manual_seed(42)
        model = CausTab(
            n_features     = data['n_features'],
            lr             = 1e-3,
            lambda_caustab = lam,
            anneal_epochs  = 50,
            random_state   = 42
        )
        # Train with fewer epochs for speed
        model.train(data['train_envs'], n_epochs=100, verbose=False)
        model.model.eval()

        aucs = []
        for env_name in ENV_ORDER:
            env_data = data['test_envs'][env_name]
            probs    = model.predict_proba(env_data['X'])
            y_true   = env_data['y'].numpy()
            auc      = roc_auc_score(y_true, probs)
            aucs.append(auc)

        mean_auc = np.mean(aucs)
        rng_auc  = max(aucs) - min(aucs)

        print(f"{lam:<10.1f} {mean_auc:>10.4f} {rng_auc:>12.4f} "
              f"{aucs[0]:>10.4f} {aucs[3]:>10.4f}")

        sensitivity_rows.append({
            'lambda':    lam,
            'mean_auc':  round(mean_auc, 4),
            'auc_range': round(rng_auc,  4),
            **{f'auc_{e}': round(a, 4)
               for e, a in zip(ENV_ORDER, aucs)}
        })

    df = pd.DataFrame(sensitivity_rows)
    df.to_csv(
        os.path.join(RESULTS_DIR, 'sensitivity_analysis.csv'), index=False
    )

    # Plot sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("CausTab — Sensitivity to λ (penalty strength)",
                fontsize=12, fontweight='bold')

    ax1 = axes[0]
    ax1.plot(df['lambda'], df['mean_auc'],
            marker='o', color='#55A868', linewidth=2)
    ax1.axvline(x=1.0, color='red', linestyle='--',
               alpha=0.6, label='λ=1.0 (used in paper)')
    ax1.set_xlabel("λ (lambda_caustab)")
    ax1.set_ylabel("Mean AUC across environments")
    ax1.set_title("Mean AUC vs λ")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2 = axes[1]
    ax2.plot(df['lambda'], df['auc_range'],
            marker='o', color='#C44E52', linewidth=2)
    ax2.axvline(x=1.0, color='red', linestyle='--',
               alpha=0.6, label='λ=1.0 (used in paper)')
    ax2.set_xlabel("λ (lambda_caustab)")
    ax2.set_ylabel("AUC range across environments")
    ax2.set_title("Stability (lower range = more stable) vs λ")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'sensitivity_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved plot: sensitivity_analysis.png")
    print(f"  Saved: sensitivity_analysis.csv")

    return df


if __name__ == "__main__":

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
    data = load_data()

    # ── Load trained models ────────────────────────────────────────────────
    print("\nLoading trained models...")
    n_features = data['n_features']

    erm     = ERM(n_features=n_features)
    irm     = IRM(n_features=n_features)
    caustab = CausTab(n_features=n_features)

    erm.model.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, 'erm_model.pt'),
                  weights_only=True))
    irm.model.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, 'irm_model.pt'),
                  weights_only=True))
    caustab.model.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, 'caustab_model.pt'),
                  weights_only=True))

    for m in [erm, irm, caustab]:
        m.model.eval()

    models_dict = {'ERM': erm, 'IRM': irm, 'CausTab': caustab}

    # ── Bootstrap CI ───────────────────────────────────────────────────────
    all_ci = run_bootstrap_analysis(
        models_dict, data['test_envs'], n_bootstrap=1000
    )
    plot_bootstrap_ci(all_ci)
    save_ci_table(all_ci)

    # ── Sensitivity analysis ───────────────────────────────────────────────
    sens_df = sensitivity_analysis(data)

    print("\n" + "="*60)
    print("SIGNIFICANCE ANALYSIS COMPLETE")
    print("Plots saved to: experiments/plots/")
    print("Results saved to: experiments/results/")
    print("="*60)