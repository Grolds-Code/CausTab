"""
CausTab — Evaluation Script
This is where the paper's main results come from.

What we measure:
- Accuracy    : fraction of correct predictions
- AUC-ROC     : Area Under the ROC Curve
                Perfect model = 1.0, random = 0.5
                Measures ability to rank hypertensive vs not
                Plain English: if I pick one hypertensive and one
                healthy person at random, AUC = probability the
                model scores the hypertensive one higher
- F1 Score    : harmonic mean of precision and recall
                Good for imbalanced outcomes (ours is 35/65)
- ECE         : Expected Calibration Error
                Measures if confidence matches reality
                ECE=0.05 means when model says 70% confident,
                true rate is between 65-75%

Key question we answer:
    Does performance DEGRADE across environments for ERM and IRM
    but STAY STABLE for CausTab?
    That degradation pattern IS the main result of the paper.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (roc_auc_score, f1_score,
                             accuracy_score, roc_curve,
                             confusion_matrix)
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data, ENV_ORDER
from models import ERM, IRM, CausTab

# ── Output directories ─────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, 'experiments', 'results')
PLOTS_DIR   = os.path.join(ROOT, 'experiments', 'plots')
TABLES_DIR  = os.path.join(ROOT, 'data', 'tables')

for d in [RESULTS_DIR, PLOTS_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

COLORS = {
    'ERM':     '#4C72B0',
    'IRM':     '#DD8452',
    'CausTab': '#55A868'
}


def compute_ece(probs, labels, n_bins=10):
    """
    Expected Calibration Error.

    Plain English:
        We divide predictions into 10 bins by confidence.
        In each bin we check: does the model's average confidence
        match the actual fraction of positive cases?
        ECE = weighted average of these gaps across bins.
        Lower = better calibrated.
    """
    bins      = np.linspace(0, 1, n_bins + 1)
    ece       = 0.0
    n_samples = len(labels)

    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc  = labels[mask].mean()
        bin_size = mask.sum()
        ece += (bin_size / n_samples) * abs(bin_conf - bin_acc)

    return ece


def evaluate_model(model, test_envs, model_name):
    """
    Evaluate a model on each environment separately.
    This reveals whether performance degrades across cycles.
    """
    results = {}

    for env_name in ENV_ORDER:
        env_data = test_envs[env_name]
        X        = env_data['X']
        y_true   = env_data['y'].numpy()

        # Get predictions
        probs  = model.predict_proba(X)
        y_pred = (probs >= 0.5).astype(int)

        # Compute all metrics
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, probs)
        f1  = f1_score(y_true, y_pred, zero_division=0)
        ece = compute_ece(probs, y_true)

        results[env_name] = {
            'accuracy': acc,
            'auc':      auc,
            'f1':       f1,
            'ece':      ece,
            'probs':    probs,
            'y_true':   y_true,
            'n':        len(y_true)
        }

    return results


def print_results_table(all_results):
    """Print a clean comparison table to terminal."""
    print("\n" + "="*75)
    print("EVALUATION RESULTS — Performance per Environment")
    print("="*75)

    for metric, label in [('auc', 'AUC-ROC'),
                           ('accuracy', 'Accuracy'),
                           ('f1', 'F1 Score'),
                           ('ece', 'ECE (↓)')]:
        print(f"\n{label}:")
        print(f"  {'Model':<12}", end="")
        for env in ENV_ORDER:
            print(f" {env:>10}", end="")
        print(f"  {'Range':>8}  {'Mean':>8}")
        print(f"  {'-'*65}")

        for model_name, env_results in all_results.items():
            vals = [env_results[e][metric] for e in ENV_ORDER]
            rng  = max(vals) - min(vals)
            mean = np.mean(vals)
            print(f"  {model_name:<12}", end="")
            for v in vals:
                print(f" {v:>10.4f}", end="")
            print(f"  {rng:>8.4f}  {mean:>8.4f}")


def save_results_tables(all_results):
    """
    Save detailed results tables as CSV and TXT.
    These go directly into the paper as Table 5.
    """
    rows = []
    for model_name, env_results in all_results.items():
        for env_name in ENV_ORDER:
            r = env_results[env_name]
            rows.append({
                'Model':       model_name,
                'Environment': env_name,
                'N':           r['n'],
                'AUC':         round(r['auc'], 4),
                'Accuracy':    round(r['accuracy'], 4),
                'F1':          round(r['f1'], 4),
                'ECE':         round(r['ece'], 4),
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, 'evaluation_results.csv')
    df.to_csv(csv_path, index=False)

    # Formatted TXT — the paper-ready version
    txt_path = os.path.join(RESULTS_DIR, 'evaluation_results.txt')
    with open(txt_path, 'w') as f:
        f.write("Table 5. Model performance by NHANES survey cycle\n")
        f.write("Metric: AUC-ROC (higher is better)\n")
        f.write("Range = max - min across cycles (lower = more stable)\n")
        f.write("="*75 + "\n")
        f.write(f"{'Model':<12}", end="", file=f) if False else None
        f.write(f"{'Model':<12}")
        for env in ENV_ORDER:
            f.write(f" {env:>10}")
        f.write(f"  {'Range':>8}  {'Mean AUC':>10}\n")
        f.write("-"*75 + "\n")

        for model_name, env_results in all_results.items():
            vals = [env_results[e]['auc'] for e in ENV_ORDER]
            rng  = max(vals) - min(vals)
            mean = np.mean(vals)
            f.write(f"{model_name:<12}")
            for v in vals:
                f.write(f" {v:>10.4f}")
            marker = " *" if model_name == 'CausTab' else "  "
            f.write(f"  {rng:>8.4f}{marker}  {mean:>8.4f}\n")

        f.write("-"*75 + "\n")
        f.write("* = proposed method\n")

    print(f"\n  Saved: evaluation_results.csv / .txt")
    return df


def plot_performance_across_environments(all_results):
    """
    The KEY plot of the paper.
    Shows AUC per environment for all three models.
    We expect ERM and IRM to degrade; CausTab to stay flat.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "CausTab — Performance Across NHANES Survey Cycles",
        fontsize=13, fontweight='bold'
    )

    x = np.arange(len(ENV_ORDER))

    # ── Plot 1: AUC per environment ────────────────────────────────────────
    ax = axes[0]
    width = 0.25
    for i, (model_name, env_results) in enumerate(all_results.items()):
        aucs = [env_results[e]['auc'] for e in ENV_ORDER]
        bars = ax.bar(x + i*width, aucs, width,
                     label=model_name,
                     color=COLORS[model_name],
                     alpha=0.85)
        # Add value labels on bars
        for bar, v in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.002,
                   f'{v:.3f}', ha='center', va='bottom',
                   fontsize=7)

    ax.set_xlabel("Survey cycle (environment)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC-ROC per environment")
    ax.set_xticks(x + width)
    ax.set_xticklabels(ENV_ORDER)
    ax.set_ylim(0.5, 0.85)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Plot 2: AUC line plot — shows trend more clearly ──────────────────
    ax2 = axes[1]
    for model_name, env_results in all_results.items():
        aucs = [env_results[e]['auc'] for e in ENV_ORDER]
        ax2.plot(ENV_ORDER, aucs,
                marker='o', linewidth=2, markersize=7,
                label=model_name,
                color=COLORS[model_name])
        # Shade the range
        ax2.fill_between(ENV_ORDER,
                        [v - 0.001 for v in aucs],
                        [v + 0.001 for v in aucs],
                        alpha=0.1, color=COLORS[model_name])

    ax2.set_xlabel("Survey cycle (environment)")
    ax2.set_ylabel("AUC-ROC")
    ax2.set_title("AUC-ROC trend across cycles\n"
                 "(flatter = more robust to distribution shift)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'performance_across_environments.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: performance_across_environments.png")


def plot_roc_curves(all_results):
    """
    ROC curves for each model on the last environment (2017-18).
    This is the hardest environment — furthest from training distribution.
    Plain English: ROC curve shows tradeoff between catching true
    hypertensives (sensitivity) vs false alarms (1-specificity).
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    target_env = '2017-18'
    for model_name, env_results in all_results.items():
        probs  = env_results[target_env]['probs']
        y_true = env_results[target_env]['y_true']
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = env_results[target_env]['auc']
        ax.plot(fpr, tpr, linewidth=2,
               color=COLORS[model_name],
               label=f"{model_name} (AUC={auc:.3f})")

    ax.plot([0,1], [0,1], 'k--', linewidth=1, alpha=0.5,
           label='Random (AUC=0.500)')
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(f"ROC Curves — Environment {target_env}\n"
                f"(most temporally distant from training)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'roc_curves_2017.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: roc_curves_2017.png")


def plot_performance_degradation(all_results):
    """
    Shows degradation relative to 2011-12 baseline.
    If a model degrades, its line drops below zero.
    CausTab should stay closest to zero.

    Plain English:
        We set 2011-12 as the reference (delta=0).
        For each later cycle, we compute how much AUC dropped.
        A flat line = robust. A dropping line = distributional fragility.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, env_results in all_results.items():
        aucs      = [env_results[e]['auc'] for e in ENV_ORDER]
        baseline  = aucs[0]
        deltas    = [a - baseline for a in aucs]
        ax.plot(ENV_ORDER, deltas,
               marker='o', linewidth=2, markersize=7,
               label=model_name,
               color=COLORS[model_name])

    ax.axhline(y=0, color='black', linestyle='--',
              linewidth=1, alpha=0.5, label='No degradation')
    ax.fill_between(ENV_ORDER,
                   [-0.005]*4, [0.005]*4,
                   alpha=0.1, color='gray',
                   label='±0.005 tolerance band')

    ax.set_xlabel("Survey cycle (environment)")
    ax.set_ylabel("ΔAUC relative to 2011-12")
    ax.set_title("Performance Degradation Across Environments\n"
                "Closer to 0 = more robust to distribution shift")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'performance_degradation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: performance_degradation.png")


def plot_feature_importance(caustab_model, data):
    """
    CausTab's learned feature importance.
    Epidemiological validation — do the learned weights
    align with known causal risk factors?

    We expect:
        HIGH importance: Age, Systolic BP, BMI, Waist
        LOW importance:  Education, Income, Race (spurious shifters)
    """
    importance = caustab_model.get_feature_importance(
        data['X_test'], data['feature_names']
    )

    # Clean feature names for display
    name_map = {
        'RIDAGEYR':  'Age',
        'RIAGENDR':  'Gender',
        'RIDRETH3':  'Race/ethnicity',
        'INDFMPIR':  'Income ratio',
        'DMDEDUC2':  'Education',
        'BPXSY1':    'Systolic BP 1',
        'BPXDI1':    'Diastolic BP 1',
        'BPXSY2':    'Systolic BP 2',
        'BPXDI2':    'Diastolic BP 2',
        'BMXBMI':    'BMI',
        'BMXWAIST':  'Waist circumference',
    }

    # Sort by importance
    items     = [(name_map[k], v) for k, v in importance.items()]
    items     = sorted(items, key=lambda x: x[1], reverse=True)
    names     = [i[0] for i in items]
    values    = [i[1] for i in items]

    # Color: known causal = green, known spurious = coral
    causal   = {'Age', 'Systolic BP 1', 'Systolic BP 2',
                'BMI', 'Waist circumference', 'Diastolic BP 1',
                'Diastolic BP 2'}
    spurious = {'Education', 'Income ratio', 'Race/ethnicity', 'Gender'}
    bar_colors = []
    for n in names:
        if n in causal:
            bar_colors.append('#55A868')
        elif n in spurious:
            bar_colors.append('#C44E52')
        else:
            bar_colors.append('#8172B2')

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, values, color=bar_colors, alpha=0.85)

    # Legend
    green_patch = mpatches.Patch(color='#55A868', label='Known causal feature')
    red_patch   = mpatches.Patch(color='#C44E52', label='Known spurious/mixed feature')
    ax.legend(handles=[green_patch, red_patch], fontsize=9)

    ax.set_xlabel("Gradient-based importance (CausTab)")
    ax.set_title("CausTab — Learned Feature Importance\n"
                "Epidemiological validation: does CausTab trust causal features?")
    ax.grid(True, alpha=0.3, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'feature_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: feature_importance.png")

    # Save as table too
    imp_df = pd.DataFrame(items, columns=['Feature', 'Importance'])
    imp_df.to_csv(os.path.join(RESULTS_DIR, 'feature_importance.csv'),
                 index=False)
    print(f"  Saved: feature_importance.csv")

    return importance


if __name__ == "__main__":

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
    data = load_data()

    # ── Reload trained models ──────────────────────────────────────────────
    # We load weights saved by train.py — no need to retrain
    print("\nLoading trained models...")
    n_features = data['n_features']

    erm     = ERM(n_features=n_features)
    irm     = IRM(n_features=n_features)
    caustab = CausTab(n_features=n_features)

    models_dir = os.path.join(ROOT, 'experiments', 'saved_models')
    erm.model.load_state_dict(
        torch.load(os.path.join(models_dir, 'erm_model.pt'),
                  weights_only=True))
    irm.model.load_state_dict(
        torch.load(os.path.join(models_dir, 'irm_model.pt'),
                  weights_only=True))
    caustab.model.load_state_dict(
        torch.load(os.path.join(models_dir, 'caustab_model.pt'),
                  weights_only=True))

    erm.model.eval()
    irm.model.eval()
    caustab.model.eval()
    print("  All models loaded.")

    # ── Evaluate all models ────────────────────────────────────────────────
    print("\nEvaluating...")
    all_results = {}
    for name, model in [('ERM', erm), ('IRM', irm), ('CausTab', caustab)]:
        all_results[name] = evaluate_model(model, data['test_envs'], name)
        print(f"  {name}: done")

    # ── Print results ──────────────────────────────────────────────────────
    print_results_table(all_results)

    # ── Save everything ────────────────────────────────────────────────────
    print("\n[Saving outputs...]")
    save_results_tables(all_results)
    plot_performance_across_environments(all_results)
    plot_roc_curves(all_results)
    plot_performance_degradation(all_results)
    plot_feature_importance(caustab, data)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("Plots saved to: experiments/plots/")
    print("Results saved to: experiments/results/")
    print("="*60)