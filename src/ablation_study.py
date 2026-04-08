"""
CausTab — Ablation Study

Tests five variants of CausTab to justify every design decision.
Each variant removes or changes exactly one component.

Why ablations matter:
    A reviewer can always say "maybe any regularization works."
    Ablations prove that each specific design choice contributes.
    If CausTab_MeanPenalty performs worse than CausTab_Full,
    that proves gradient VARIANCE specifically matters — not
    just any gradient-based penalty.

Variants tested:
    Full        — complete CausTab (our method)
    NoAnneal    — no annealing, penalty active from epoch 1
    NoWarmup    — penalty jumps to full strength at epoch 50
                  no linear ramp
    MeanPenalty — use gradient MEAN instead of VARIANCE
                  tests whether variance specifically matters
    NoPenalty   — pure ERM, no invariance penalty
                  establishes that penalty contributes

We run ablations on:
    1. All three synthetic regimes — controlled ground truth
    2. NHANES temporal split B — real data validation
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import Network, bce_loss, irm_penalty
from synthetic_experiment import generate_dataset, REGIMES, CONFIG
from data_loader import load_data, ENV_ORDER
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, 'experiments', 'results', 'ablation')
PLOTS_DIR   = os.path.join(ROOT, 'experiments', 'plots', 'ablation')

for d in [RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

COLORS = {
    'CausTab_Full':        '#55A868',
    'CausTab_NoAnneal':    '#4C72B0',
    'CausTab_NoWarmup':    '#8172B2',
    'CausTab_MeanPenalty': '#CCB974',
    'CausTab_NoPenalty':   '#C44E52',
}

VARIANT_LABELS = {
    'CausTab_Full':        'CausTab (full)',
    'CausTab_NoAnneal':    'No annealing',
    'CausTab_NoWarmup':    'No warmup ramp',
    'CausTab_MeanPenalty': 'Mean penalty\n(not variance)',
    'CausTab_NoPenalty':   'No penalty\n(= ERM)',
}


class CausTabVariant:
    """
    Flexible CausTab implementation that supports
    all five ablation variants through config flags.

    Plain English:
        Instead of five separate classes, we use one class
        with switches that turn components on and off.
        This ensures the only difference between variants
        is exactly the component being ablated.
    """

    def __init__(self, n_features=11, lr=1e-3,
                 lambda_caustab=1.0,
                 anneal_epochs=50,
                 use_anneal=True,
                 use_warmup=True,
                 use_variance=True,
                 use_penalty=True,
                 random_state=42,
                 name='CausTab_Full'):

        torch.manual_seed(random_state)
        self.model           = Network(n_features=n_features)
        self.optimizer       = torch.optim.Adam(
            self.model.parameters(), lr=lr)
        self.lambda_caustab  = lambda_caustab
        self.anneal_epochs   = anneal_epochs
        self.use_anneal      = use_anneal
        self.use_warmup      = use_warmup
        self.use_variance    = use_variance
        self.use_penalty     = use_penalty
        self.name            = name
        self.train_losses    = []
        self.penalty_history = []

    def train(self, train_envs, n_epochs=200, verbose=False):
        env_list = list(train_envs.values())

        for epoch in range(n_epochs):
            self.model.train()
            self.optimizer.zero_grad()

            # ── ERM component ──────────────────────────────────────────────
            total_erm = 0.0
            for env_data in env_list:
                preds    = self.model(env_data['X'])
                env_loss = bce_loss(preds, env_data['y'])
                total_erm += env_loss
            total_erm /= len(env_list)

            # ── Penalty component ──────────────────────────────────────────
            penalty        = torch.tensor(0.0)
            penalty_weight = 0.0

            if self.use_penalty:

                # Determine penalty weight based on variant
                if not self.use_anneal:
                    # NoAnneal: penalty active from epoch 1
                    penalty_weight = self.lambda_caustab

                elif not self.use_warmup:
                    # NoWarmup: penalty jumps at anneal_epochs
                    if epoch >= self.anneal_epochs:
                        penalty_weight = self.lambda_caustab

                else:
                    # Full: anneal + linear warmup
                    if epoch >= self.anneal_epochs:
                        warmup_epochs  = 20
                        epochs_past    = epoch - self.anneal_epochs
                        ramp           = min(1.0,
                                            epochs_past / warmup_epochs)
                        penalty_weight = self.lambda_caustab * ramp

                # Compute penalty if weight > 0
                if penalty_weight > 0:
                    env_gradients = []
                    for env_data in env_list:
                        preds_e  = self.model(env_data['X'])
                        loss_e   = bce_loss(preds_e, env_data['y'])
                        grads    = torch.autograd.grad(
                            loss_e,
                            self.model.parameters(),
                            create_graph=True,
                            retain_graph=True
                        )
                        grad_vec = torch.cat(
                            [g.reshape(-1) for g in grads])
                        env_gradients.append(grad_vec)

                    grad_matrix = torch.stack(env_gradients)

                    if self.use_variance:
                        # Full CausTab: gradient VARIANCE
                        penalty = grad_matrix.var(dim=0).mean()
                    else:
                        # MeanPenalty: gradient MEAN magnitude
                        # Tests whether variance specifically matters
                        penalty = grad_matrix.mean(dim=0).abs().mean()

            # ── Combined loss ──────────────────────────────────────────────
            loss = total_erm + penalty_weight * penalty
            loss.backward()
            self.optimizer.step()

            self.train_losses.append(total_erm.item())
            self.penalty_history.append(
                penalty.item()
                if isinstance(penalty, torch.Tensor)
                else 0.0
            )

        self.model.eval()
        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X).numpy()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def get_variants(n_features, config):
    """
    Instantiate all five ablation variants.
    Returns a dict of name → model.
    """
    return {
        'CausTab_Full': CausTabVariant(
            n_features     = n_features,
            lr             = config['lr'],
            lambda_caustab = config['lambda_caustab'],
            anneal_epochs  = config['anneal_epochs'],
            use_anneal     = True,
            use_warmup     = True,
            use_variance   = True,
            use_penalty    = True,
            random_state   = config['random_state'],
            name           = 'CausTab_Full'
        ),
        'CausTab_NoAnneal': CausTabVariant(
            n_features     = n_features,
            lr             = config['lr'],
            lambda_caustab = config['lambda_caustab'],
            anneal_epochs  = config['anneal_epochs'],
            use_anneal     = False,
            use_warmup     = False,
            use_variance   = True,
            use_penalty    = True,
            random_state   = config['random_state'],
            name           = 'CausTab_NoAnneal'
        ),
        'CausTab_NoWarmup': CausTabVariant(
            n_features     = n_features,
            lr             = config['lr'],
            lambda_caustab = config['lambda_caustab'],
            anneal_epochs  = config['anneal_epochs'],
            use_anneal     = True,
            use_warmup     = False,
            use_variance   = True,
            use_penalty    = True,
            random_state   = config['random_state'],
            name           = 'CausTab_NoWarmup'
        ),
        'CausTab_MeanPenalty': CausTabVariant(
            n_features     = n_features,
            lr             = config['lr'],
            lambda_caustab = config['lambda_caustab'],
            anneal_epochs  = config['anneal_epochs'],
            use_anneal     = True,
            use_warmup     = True,
            use_variance   = False,
            use_penalty    = True,
            random_state   = config['random_state'],
            name           = 'CausTab_MeanPenalty'
        ),
        'CausTab_NoPenalty': CausTabVariant(
            n_features     = n_features,
            lr             = config['lr'],
            lambda_caustab = config['lambda_caustab'],
            anneal_epochs  = config['anneal_epochs'],
            use_anneal     = True,
            use_warmup     = True,
            use_variance   = True,
            use_penalty    = False,
            random_state   = config['random_state'],
            name           = 'CausTab_NoPenalty'
        ),
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


def run_ablation_synthetic(n_seeds=5):
    """
    Run ablation on synthetic data across all three regimes.
    Uses multiple seeds for robust estimates.
    """
    print("\n[Synthetic Ablation]")
    all_results = {}

    for regime_name, regime_cfg in REGIMES.items():
        print(f"\n  Regime: {regime_cfg['description']}")
        all_results[regime_name] = {
            v: [] for v in [
                'CausTab_Full', 'CausTab_NoAnneal',
                'CausTab_NoWarmup', 'CausTab_MeanPenalty',
                'CausTab_NoPenalty'
            ]
        }

        for seed in range(n_seeds):
            dataset  = generate_dataset(regime_cfg, CONFIG, seed=seed)
            variants = get_variants(dataset['n_features'], CONFIG)

            for var_name, model in variants.items():
                model.train(dataset['train_envs'],
                           n_epochs=CONFIG['n_epochs'],
                           verbose=False)

                all_probs, all_y = [], []
                for env_data in dataset['test_envs'].values():
                    probs  = model.predict_proba(env_data['X'])
                    y_true = env_data['y'].numpy()
                    all_probs.append(probs)
                    all_y.append(y_true)

                auc = roc_auc_score(
                    np.concatenate(all_y),
                    np.concatenate(all_probs)
                )
                all_results[regime_name][var_name].append(auc)

        # Print regime summary
        print(f"  {'Variant':<25} {'Mean AUC':>10} "
              f"{'Std':>8} {'vs Full':>10}")
        print(f"  {'-'*55}")
        full_mean = np.mean(
            all_results[regime_name]['CausTab_Full'])
        for var_name in all_results[regime_name]:
            aucs    = all_results[regime_name][var_name]
            mean    = np.mean(aucs)
            std     = np.std(aucs)
            vs_full = mean - full_mean
            marker  = " ★" if var_name == 'CausTab_Full' else ""
            print(f"  {var_name:<25} {mean:>10.4f} "
                  f"{std:>8.4f} {vs_full:>+10.4f}{marker}")

    return all_results


def run_ablation_nhanes(n_seeds=3):
    """
    Run ablation on NHANES temporal Split B.
    Validates that ablation findings hold on real data.
    """
    print("\n[NHANES Ablation — Temporal Split B]")
    print("  Train: 2011-12, 2013-14 | Test: 2015-16, 2017-18")

    df       = pd.read_csv(
        os.path.join(ROOT, 'data', 'nhanes_master.csv'))
    FEATURE_COLS = [
        'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'INDFMPIR',
        'DMDEDUC2', 'BPXSY1', 'BPXDI1', 'BPXSY2',
        'BPXDI2', 'BMXBMI', 'BMXWAIST',
    ]
    TRAIN_ENVS = ['2011-12', '2013-14']
    TEST_ENVS  = ['2015-16', '2017-18']

    nhanes_results = {
        v: {'auc': [], 'ece': []}
        for v in [
            'CausTab_Full', 'CausTab_NoAnneal',
            'CausTab_NoWarmup', 'CausTab_MeanPenalty',
            'CausTab_NoPenalty'
        ]
    }

    for seed in range(n_seeds):
        # Prepare data
        train_mask = df['environment'].isin(TRAIN_ENVS)
        test_mask  = df['environment'].isin(TEST_ENVS)

        X_train_raw = df.loc[train_mask, FEATURE_COLS].values.astype(
            np.float32)
        X_test_raw  = df.loc[test_mask,  FEATURE_COLS].values.astype(
            np.float32)
        y_train = df.loc[train_mask, 'hypertension'].values.astype(
            np.float32)
        y_test  = df.loc[test_mask,  'hypertension'].values.astype(
            np.float32)
        env_train = df.loc[train_mask, 'environment'].values
        env_test  = df.loc[test_mask,  'environment'].values

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_test  = scaler.transform(X_test_raw).astype(np.float32)

        # Build per-environment dicts
        train_envs_dict = {}
        for e in TRAIN_ENVS:
            mask = env_train == e
            train_envs_dict[e] = {
                'X': torch.FloatTensor(X_train[mask]),
                'y': torch.FloatTensor(y_train[mask]),
                'n': int(mask.sum())
            }

        nhanes_config = {
            'lr':              1e-3,
            'lambda_caustab':  1.0,
            'anneal_epochs':   50,
            'random_state':    seed,
        }

        variants = get_variants(len(FEATURE_COLS), nhanes_config)

        for var_name, model in variants.items():
            model.train(train_envs_dict,
                       n_epochs=200,
                       verbose=False)

            probs  = model.predict_proba(
                torch.FloatTensor(X_test))
            y_pred = (probs >= 0.5).astype(int)

            auc = roc_auc_score(y_test, probs)
            ece = compute_ece(probs, y_test)

            nhanes_results[var_name]['auc'].append(auc)
            nhanes_results[var_name]['ece'].append(ece)

    # Print summary
    print(f"\n  {'Variant':<25} {'Mean AUC':>10} "
          f"{'Mean ECE':>10} {'vs Full AUC':>12}")
    print(f"  {'-'*60}")
    full_auc = np.mean(nhanes_results['CausTab_Full']['auc'])
    for var_name, res in nhanes_results.items():
        mean_auc = np.mean(res['auc'])
        mean_ece = np.mean(res['ece'])
        vs_full  = mean_auc - full_auc
        marker   = " ★" if var_name == 'CausTab_Full' else ""
        print(f"  {var_name:<25} {mean_auc:>10.4f} "
              f"{mean_ece:>10.4f} {vs_full:>+12.4f}{marker}")

    return nhanes_results


def plot_ablation_synthetic(all_results):
    """
    Bar chart showing all five variants across three regimes.
    CausTab_Full should be best or tied-best in every regime.
    """
    variant_names = [
        'CausTab_Full', 'CausTab_NoAnneal',
        'CausTab_NoWarmup', 'CausTab_MeanPenalty',
        'CausTab_NoPenalty'
    ]
    regime_names  = list(REGIMES.keys())
    n_variants    = len(variant_names)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "CausTab Ablation Study — Synthetic Data\n"
        "Each variant removes exactly one component",
        fontsize=12, fontweight='bold'
    )

    for ax, regime_name in zip(axes, regime_names):
        means  = [np.mean(all_results[regime_name][v])
                 for v in variant_names]
        stds   = [np.std(all_results[regime_name][v])
                 for v in variant_names]
        colors = [COLORS[v] for v in variant_names]
        x      = np.arange(n_variants)

        bars = ax.bar(x, means, color=colors, alpha=0.85,
                     yerr=stds, capsize=5,
                     error_kw={'linewidth': 1.2})

        # Highlight full method
        bars[0].set_edgecolor('black')
        bars[0].set_linewidth(2)

        # Value labels
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + s + 0.003,
                   f'{m:.3f}',
                   ha='center', va='bottom',
                   fontsize=7, fontweight='bold')

        regime_cfg  = REGIMES[regime_name]
        short       = regime_name.replace(
            'Regime_', 'R').replace('_', ' ')
        ax.set_title(f"{short}\n{regime_cfg['description']}",
                    fontsize=9)
        labels = [VARIANT_LABELS[v].replace('\n', ' ')
                 for v in variant_names]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=20,
                          ha='right')
        ax.set_ylabel("Mean AUC-ROC")
        ax.set_ylim(min(means) - 0.05, max(means) + 0.05)
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'ablation_synthetic.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: ablation_synthetic.png")


def plot_ablation_nhanes(nhanes_results):
    """
    Bar chart showing ablation results on NHANES.
    Two metrics: AUC and ECE.
    """
    variant_names = [
        'CausTab_Full', 'CausTab_NoAnneal',
        'CausTab_NoWarmup', 'CausTab_MeanPenalty',
        'CausTab_NoPenalty'
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "CausTab Ablation Study — NHANES Temporal Split B\n"
        "Real epidemiological data validation",
        fontsize=12, fontweight='bold'
    )

    for ax, metric, ylabel, title in zip(
        axes,
        ['auc', 'ece'],
        ['Mean AUC-ROC', 'Mean ECE (↓ better)'],
        ['AUC on NHANES temporal split',
         'Calibration (ECE) on NHANES temporal split']
    ):
        means  = [np.mean(nhanes_results[v][metric])
                 for v in variant_names]
        stds   = [np.std(nhanes_results[v][metric])
                 for v in variant_names]
        colors = [COLORS[v] for v in variant_names]
        x      = np.arange(len(variant_names))

        bars = ax.bar(x, means, color=colors, alpha=0.85,
                     yerr=stds, capsize=5,
                     error_kw={'linewidth': 1.2})

        bars[0].set_edgecolor('black')
        bars[0].set_linewidth(2)

        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + s + 0.001,
                   f'{m:.4f}',
                   ha='center', va='bottom', fontsize=8)

        labels = [VARIANT_LABELS[v].replace('\n', ' ')
                 for v in variant_names]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8,
                          rotation=20, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        rng = max(means) - min(means)
        ax.set_ylim(min(means) - rng*0.5,
                   max(means) + rng*0.5)
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'ablation_nhanes.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: ablation_nhanes.png")


def save_ablation_results(synthetic_results,
                          nhanes_results):
    """Save all ablation results."""
    # Synthetic results
    rows = []
    for regime_name in REGIMES:
        for var_name in synthetic_results[regime_name]:
            aucs = synthetic_results[regime_name][var_name]
            rows.append({
                'Regime':    regime_name,
                'Variant':   var_name,
                'Mean_AUC':  round(np.mean(aucs), 4),
                'Std_AUC':   round(np.std(aucs),  4),
            })
    syn_df = pd.DataFrame(rows)
    syn_df.to_csv(
        os.path.join(RESULTS_DIR,
                    'ablation_synthetic_results.csv'),
        index=False
    )

    # NHANES results
    nhanes_rows = []
    for var_name, res in nhanes_results.items():
        nhanes_rows.append({
            'Variant':  var_name,
            'Mean_AUC': round(np.mean(res['auc']), 4),
            'Std_AUC':  round(np.std(res['auc']),  4),
            'Mean_ECE': round(np.mean(res['ece']), 4),
            'Std_ECE':  round(np.std(res['ece']),  4),
        })
    nhanes_df = pd.DataFrame(nhanes_rows)
    nhanes_df.to_csv(
        os.path.join(RESULTS_DIR,
                    'ablation_nhanes_results.csv'),
        index=False
    )

    # Paper-ready TXT
    txt_path = os.path.join(
        RESULTS_DIR, 'ablation_summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("CausTab Ablation Study — Summary\n")
        f.write("="*65 + "\n\n")
        f.write("Synthetic Data (mean AUC ± std, 5 seeds)\n")
        f.write("-"*65 + "\n")
        f.write(f"{'Variant':<25}")
        for r in REGIMES:
            short = r.replace('Regime_', 'R').replace(
                '_Causal_Dominant', '1').replace(
                '_Mixed', '2').replace(
                '_Spurious_Dominant', '3')
            f.write(f" {short:>18}")
        f.write("\n" + "-"*65 + "\n")
        for var_name in [
            'CausTab_Full', 'CausTab_NoAnneal',
            'CausTab_NoWarmup', 'CausTab_MeanPenalty',
            'CausTab_NoPenalty'
        ]:
            f.write(f"{var_name:<25}")
            for regime_name in REGIMES:
                aucs = synthetic_results[regime_name][var_name]
                cell = f"{np.mean(aucs):.3f}±{np.std(aucs):.3f}"
                marker = "★" if var_name == 'CausTab_Full' else " "
                f.write(f" {cell:>17}{marker}")
            f.write("\n")
        f.write("\nNHANES Temporal Split B\n")
        f.write("-"*65 + "\n")
        f.write(f"{'Variant':<25} {'AUC':>12} {'ECE':>12}\n")
        f.write("-"*65 + "\n")
        for var_name, res in nhanes_results.items():
            marker = " ★" if var_name == 'CausTab_Full' else "  "
            f.write(
                f"{var_name:<25} "
                f"{np.mean(res['auc']):>10.4f}  "
                f"{np.mean(res['ece']):>10.4f}{marker}\n"
            )
        f.write("\n★ = proposed full method\n")

    print(f"  Saved: ablation_synthetic_results.csv")
    print(f"  Saved: ablation_nhanes_results.csv")
    print(f"  Saved: ablation_summary.txt")

    return syn_df, nhanes_df


if __name__ == "__main__":
    print("="*65)
    print("CausTab — ABLATION STUDY")
    print("="*65)
    print("Variants: 5")
    print("Synthetic: 3 regimes × 5 seeds")
    print("NHANES: temporal split B × 3 seeds")
    print("="*65)

    # ── Synthetic ablation ─────────────────────────────────────────────────
    print("\n[1/3] Running synthetic ablation...")
    start           = time.time()
    synthetic_results = run_ablation_synthetic(n_seeds=5)
    print(f"  Done in {time.time()-start:.1f}s")

    # ── NHANES ablation ────────────────────────────────────────────────────
    print("\n[2/3] Running NHANES ablation...")
    start          = time.time()
    nhanes_results = run_ablation_nhanes(n_seeds=3)
    print(f"  Done in {time.time()-start:.1f}s")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n[3/3] Generating plots...")
    plot_ablation_synthetic(synthetic_results)
    plot_ablation_nhanes(nhanes_results)

    # ── Save ───────────────────────────────────────────────────────────────
    print("\nSaving results...")
    save_ablation_results(synthetic_results, nhanes_results)

    print(f"\n{'='*65}")
    print("ABLATION STUDY COMPLETE")
    print(f"Plots:   experiments/plots/ablation/")
    print(f"Results: experiments/results/ablation/")
    print(f"{'='*65}")