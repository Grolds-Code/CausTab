"""
CausTab — Training Script
Trains all three models and saves results.

What this file does:
- Loads data via data_loader
- Trains ERM, IRM, and CausTab
- Saves trained models to disk
- Saves training curves (loss over epochs)
- Reports training time per model

Plain English — what is training?
    Training is the process of adjusting model weights to minimize
    the loss function. We pass data through the network, compute
    how wrong the predictions are (loss), then use backpropagation
    to figure out which weights caused the error, then update those
    weights slightly in the direction that reduces error.
    We repeat this hundreds of times until the model stabilizes.
    Each full pass through the training logic is called an epoch.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time

# Add src/ to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data
from models import ERM, IRM, CausTab

# ── Output directories ─────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, 'experiments', 'results')
MODELS_DIR  = os.path.join(ROOT, 'experiments', 'saved_models')
PLOTS_DIR   = os.path.join(ROOT, 'experiments', 'plots')

for d in [RESULTS_DIR, MODELS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
# Hyperparameters = settings we choose before training, not learned
# Plain English: these are the knobs we turn to control training

CONFIG = {
    'n_epochs':        200,   # How many times we pass through the training logic
    'lr':              1e-3,  # Learning rate — how big each weight update step is
                              # Too high = overshoots, too low = too slow
    'lambda_irm':      1.0,   # IRM penalty strength
    'lambda_caustab':  1.0,   # CausTab penalty strength
    'anneal_epochs':   50,    # CausTab: epochs before penalty kicks in
    'random_state':    42,    # Random seed — ensures reproducibility
                              # Same seed = same results every run
}


def train_all_models(data, config=CONFIG, verbose=True):
    """
    Train ERM, IRM, and CausTab on the same data.
    Returns all three trained models and their training histories.
    """
    print("\n" + "="*60)
    print("CAUSTAB — TRAINING")
    print("="*60)
    print(f"Epochs: {config['n_epochs']} | LR: {config['lr']}")
    print(f"IRM lambda: {config['lambda_irm']} | "
          f"CausTab lambda: {config['lambda_caustab']}")
    print(f"CausTab anneal epochs: {config['anneal_epochs']}")
    print("="*60)

    train_envs = data['train_envs']
    n_features = data['n_features']
    results    = {}

    # ── Train ERM ──────────────────────────────────────────────────────────
    print(f"\n[1/3] Training ERM (baseline)...")
    start = time.time()
    erm   = ERM(
        n_features   = n_features,
        lr           = config['lr'],
        random_state = config['random_state']
    )
    erm.train(train_envs, n_epochs=config['n_epochs'], verbose=verbose)
    erm_time = time.time() - start
    print(f"  Done in {erm_time:.1f}s")
    results['ERM'] = {'model': erm, 'time': erm_time}

    # ── Train IRM ──────────────────────────────────────────────────────────
    print(f"\n[2/3] Training IRM (existing method)...")
    start = time.time()
    irm   = IRM(
        n_features   = n_features,
        lr           = config['lr'],
        lambda_irm   = config['lambda_irm'],
        random_state = config['random_state']
    )
    irm.train(train_envs, n_epochs=config['n_epochs'], verbose=verbose)
    irm_time = time.time() - start
    print(f"  Done in {irm_time:.1f}s")
    results['IRM'] = {'model': irm, 'time': irm_time}

    # ── Train CausTab ──────────────────────────────────────────────────────
    print(f"\n[3/3] Training CausTab (our method)...")
    start    = time.time()
    caustab  = CausTab(
        n_features      = n_features,
        lr              = config['lr'],
        lambda_caustab  = config['lambda_caustab'],
        anneal_epochs   = config['anneal_epochs'],
        random_state    = config['random_state']
    )
    caustab.train(train_envs, n_epochs=config['n_epochs'], verbose=verbose)
    caustab_time = time.time() - start
    print(f"  Done in {caustab_time:.1f}s")
    results['CausTab'] = {'model': caustab, 'time': caustab_time}

    return results


def save_models(results):
    """
    Save trained model weights to disk.

    Plain English:
        torch.save stores the model's learned weights as a file.
        This means we can load the model later without retraining.
        state_dict() = dictionary of all parameter tensors.
    """
    print("\n[Saving models...]")
    for name, res in results.items():
        path = os.path.join(MODELS_DIR, f"{name.lower()}_model.pt")
        torch.save(res['model'].model.state_dict(), path)
        print(f"  Saved: {path}")


def plot_training_curves(results):
    """
    Plot training loss over epochs for all three models.
    This goes in the paper as evidence that training was stable.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("CausTab — Training Loss Curves", fontsize=13)

    colors = {'ERM': '#4C72B0', 'IRM': '#DD8452', 'CausTab': '#55A868'}

    for ax, (name, res) in zip(axes, results.items()):
        model  = res['model']
        losses = model.train_losses
        epochs = range(1, len(losses) + 1)

        ax.plot(epochs, losses, color=colors[name], linewidth=1.5)
        ax.set_title(f"{name}", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Mark anneal point for CausTab
        if name == 'CausTab' and hasattr(model, 'anneal_epochs'):
            ax.axvline(x=model.anneal_epochs, color='red',
                      linestyle='--', alpha=0.6, linewidth=1)
            ax.text(model.anneal_epochs + 2,
                   max(losses) * 0.95,
                   'Penalty\nstarts',
                   fontsize=8, color='red', alpha=0.8)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {path}")


def plot_caustab_penalty(results):
    """
    Plot CausTab's gradient variance penalty over training.
    Shows the penalty ramping up after annealing period.
    This is a key diagnostic plot for the paper.
    """
    caustab = results['CausTab']['model']
    penalty = caustab.penalty_history
    epochs  = range(1, len(penalty) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, penalty, color='#55A868', linewidth=1.5)
    ax.axvline(x=caustab.anneal_epochs, color='red',
              linestyle='--', alpha=0.6, linewidth=1)
    ax.text(caustab.anneal_epochs + 2,
           max(penalty) * 0.9,
           'Penalty active',
           fontsize=9, color='red')
    ax.set_title("CausTab — Gradient Variance Penalty During Training",
                fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Variance Penalty")
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'caustab_penalty.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {path}")


def save_training_summary(results, config):
    """
    Save a summary of training to CSV and TXT.
    """
    rows = []
    for name, res in results.items():
        rows.append({
            'Model':           name,
            'Epochs':          config['n_epochs'],
            'Learning_rate':   config['lr'],
            'Final_loss':      round(res['model'].train_losses[-1], 4),
            'Training_time_s': round(res['time'], 1),
        })

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, 'training_summary.csv')
    df.to_csv(csv_path, index=False)

    # Save TXT
    txt_path = os.path.join(RESULTS_DIR, 'training_summary.txt')
    with open(txt_path, 'w') as f:
        f.write("CausTab — Training Summary\n")
        f.write("="*55 + "\n")
        f.write(f"{'Model':<12} {'Epochs':>8} {'Final Loss':>12} "
                f"{'Time (s)':>10}\n")
        f.write("-"*55 + "\n")
        for _, row in df.iterrows():
            f.write(f"{row['Model']:<12} {int(row['Epochs']):>8} "
                    f"{row['Final_loss']:>12.4f} "
                    f"{row['Training_time_s']:>10.1f}\n")

    print(f"  Saved: training_summary.csv / .txt")
    print(f"\n{df.to_string(index=False)}")

    return df


if __name__ == "__main__":
    # ── Load data ──────────────────────────────────────────────────────────
    data = load_data()

    # ── Train all models ───────────────────────────────────────────────────
    results = train_all_models(data, verbose=True)

    # ── Save everything ────────────────────────────────────────────────────
    print("\n[Saving outputs...]")
    save_models(results)
    plot_training_curves(results)
    plot_caustab_penalty(results)
    save_training_summary(results, CONFIG)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"Models saved to:  experiments/saved_models/")
    print(f"Plots saved to:   experiments/plots/")
    print(f"Results saved to: experiments/results/")
    print("="*60)
    print("\nNext step: run src/evaluate.py")