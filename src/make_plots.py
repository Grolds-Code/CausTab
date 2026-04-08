"""
CausTab — Publication Quality Plot Generator

Regenerates all paper figures with consistent styling.
Style: Clean academic (Nature/ICML standard)

Design principles:
    - Generous whitespace — nothing cramped
    - Font size minimum 10pt everywhere
    - No overlapping text ever
    - Consistent color palette throughout
    - Maximum 300 DPI for print quality
    - All text horizontal — no rotated labels
    - Error bars always shown where data supports them
    - Legends outside plot area when space is tight
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import os
import sys

# ── Global style settings ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          11,
    'axes.titlesize':     13,
    'axes.labelsize':     11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    10,
    'figure.titlesize':   14,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'grid.linewidth':     0.5,
    'lines.linewidth':    2.0,
    'patch.linewidth':    0.8,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.15,
})

# ── Color palette — consistent throughout all figures ─────────────────────────
COLORS = {
    'ERM':     '#2166AC',   # blue
    'IRM':     '#D6604D',   # red-orange
    'CausTab': '#1A9641',   # green
}

REGIME_COLORS = {
    'Regime_1_Causal_Dominant':   '#4DAC26',
    'Regime_2_Mixed':             '#F1A340',
    'Regime_3_Spurious_Dominant': '#D01C8B',
}

HOSPITAL_COLORS = {
    'Cleveland':    '#4C72B0',
    'Hungary':      '#DD8452',
    'Switzerland':  '#55A868',
    'VA Long Beach':'#C44E52',
}

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_ROOT = os.path.join(ROOT, 'experiments', 'plots', 'publication')
os.makedirs(PLOT_ROOT, exist_ok=True)

def save(fig, name):
    path = os.path.join(PLOT_ROOT, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Distribution shift evidence (NHANES exploration)
# ══════════════════════════════════════════════════════════════════════════════

def fig1_nhanes_shift_evidence():
    """
    Shows correlation shift across NHANES cycles.
    The smoking gun that motivates the paper.
    """
    csv = os.path.join(ROOT, 'data', 'tables',
                       'table3_correlation_shift.csv')
    if not os.path.exists(csv):
        print("  SKIP fig1 — table3_correlation_shift.csv not found")
        return

    df = pd.read_csv(csv)

    # Clean feature names
    name_map = {
        'Age (years)':              'Age',
        'Gender (1=M, 2=F)':        'Gender',
        'Race/ethnicity':           'Race/ethnicity',
        'Income-to-poverty ratio':  'Income ratio',
        'Education level':          'Education',
        'Systolic BP 1 (mmHg)':     'Systolic BP 1',
        'Diastolic BP 1 (mmHg)':    'Diastolic BP 1',
        'Systolic BP 2 (mmHg)':     'Systolic BP 2',
        'Diastolic BP 2 (mmHg)':    'Diastolic BP 2',
        'BMI (kg/m²)':              'BMI',
        'Waist circumference (cm)': 'Waist circ.',
    }
    df['Feature'] = df['Feature'].map(
        lambda x: name_map.get(x, x))

    cycles = ['2011-12', '2013-14', '2015-16', '2017-18']
    available = [c for c in cycles if c in df.columns]

    # Sort by range descending
    df = df.sort_values('Range', ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Figure 1. Distribution Shift Evidence in NHANES\n"
        "Feature–outcome correlations vary across survey cycles",
        fontweight='bold', y=1.02
    )

    # Left: heatmap of correlations
    ax = axes[0]
    corr_matrix = df.set_index('Feature')[available].values
    features    = df['Feature'].tolist()

    im = ax.imshow(corr_matrix, cmap='RdBu_r',
                  vmin=-0.5, vmax=0.5, aspect='auto')
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels(available, rotation=0)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)
    ax.set_title("Correlation with hypertension\nper survey cycle")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                label='Pearson correlation')

    # Add text annotations
    for i in range(len(features)):
        for j in range(len(available)):
            val = corr_matrix[i, j]
            ax.text(j, i, f'{val:.2f}',
                   ha='center', va='center',
                   fontsize=8,
                   color='white' if abs(val) > 0.3 else 'black')

    # Right: range bar chart
    ax2 = axes[1]
    ranges = df['Range'].values
    colors_bar = ['#C44E52' if r > 0.02 else '#55A868'
                 for r in ranges]
    bars = ax2.barh(features, ranges,
                   color=colors_bar, alpha=0.85,
                   height=0.6)
    ax2.axvline(x=0.02, color='gray', linestyle='--',
               linewidth=1, alpha=0.7,
               label='Threshold (0.02)')
    ax2.set_xlabel("Correlation range\n(max − min across cycles)")
    ax2.set_title("Feature stability\nacross NHANES cycles")
    ax2.legend(loc='lower right', fontsize=9)

    # Legend for colors
    stable_patch   = mpatches.Patch(
        color='#55A868', label='Stable (causal)')
    unstable_patch = mpatches.Patch(
        color='#C44E52', label='Unstable (spurious/mixed)')
    ax2.legend(handles=[stable_patch, unstable_patch],
              loc='lower right', fontsize=9)

    fig.tight_layout()
    save(fig, 'fig1_nhanes_shift_evidence.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — IRM failure analysis (the centerpiece)
# ══════════════════════════════════════════════════════════════════════════════

def fig2_irm_failure():
    """
    Shows IRM degradation across spurious strengths.
    Most important result figure in the paper.
    """
    csv = os.path.join(ROOT, 'experiments', 'results',
                       'irm_analysis', 'spurious_sweep_results.csv')
    if not os.path.exists(csv):
        print("  SKIP fig2 — spurious_sweep_results.csv not found")
        return

    df = pd.read_csv(csv)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Figure 2. IRM Failure on Tabular Data\n"
        "Performance vs spurious feature strength "
        "(complete shift at test time)",
        fontweight='bold', y=1.02
    )

    strengths = sorted(df['Spurious_strength'].unique())

    for ax, metric, ylabel, title in zip(
        axes,
        ['Mean_AUC', 'Std_AUC'],
        ['Mean AUC-ROC (± std across seeds)',
         'Std of AUC across seeds\n(lower = more stable)'],
        ['Predictive performance vs spurious strength',
         'Stability vs spurious strength']
    ):
        for model_name in ['ERM', 'IRM', 'CausTab']:
            sub   = df[df['Model'] == model_name]
            vals  = [sub[sub['Spurious_strength']==s][metric].values[0]
                    for s in strengths]
            ax.plot(strengths, vals,
                   marker='o', markersize=7,
                   color=COLORS[model_name],
                   label=model_name,
                   zorder=3)

            if metric == 'Mean_AUC':
                stds = [sub[sub['Spurious_strength']==s
                           ]['Std_AUC'].values[0]
                       for s in strengths]
                ax.fill_between(
                    strengths,
                    [v-s for v,s in zip(vals,stds)],
                    [v+s for v,s in zip(vals,stds)],
                    alpha=0.12,
                    color=COLORS[model_name]
                )

        ax.set_xlabel("Spurious feature strength in training",
                     fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, pad=10)
        ax.legend(loc='lower left' if metric=='Mean_AUC'
                 else 'upper left',
                 framealpha=0.9)

        if metric == 'Mean_AUC':
            ax.set_ylim(0.45, 1.02)
            ax.axhline(y=0.5, color='gray',
                      linestyle=':', linewidth=1,
                      alpha=0.7, label='Random')

    fig.tight_layout()
    save(fig, 'fig2_irm_failure.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Synthetic regime comparison
# ══════════════════════════════════════════════════════════════════════════════

def fig3_synthetic_regimes():
    """
    Three regime comparison — proves CausTab works
    and IRM fails across controlled conditions.
    """
    csv = os.path.join(ROOT, 'experiments', 'results',
                       'synthetic', 'synthetic_results.csv')
    if not os.path.exists(csv):
        print("  SKIP fig3 — synthetic_results.csv not found")
        return

    df = pd.read_csv(csv)

    regimes = df['Regime'].unique()
    regime_labels = {
        'Regime_1_Causal_Dominant':
            'Regime 1\nCausal dominant\n(SDI = 1.67)',
        'Regime_2_Mixed':
            'Regime 2\nMixed\n(SDI = 9.62)',
        'Regime_3_Spurious_Dominant':
            'Regime 3\nSpurious dominant\n(SDI = 48.08)',
    }

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle(
        "Figure 3. Synthetic Experiment — Three Spurious Regimes\n"
        "Mean AUC ± std across 5 random seeds",
        fontweight='bold', y=1.02
    )

    n_regimes = len(regimes)
    n_models  = 3
    width     = 0.22
    x         = np.arange(n_regimes)

    for i, model_name in enumerate(['ERM', 'IRM', 'CausTab']):
        sub   = df[df['Model'] == model_name]
        means = [sub[sub['Regime']==r]['Mean_AUC'].values[0]
                for r in regimes]
        stds  = [sub[sub['Regime']==r]['Std_AUC'].values[0]
                for r in regimes]

        offset = (i - 1) * (width + 0.02)
        bars   = ax.bar(x + offset, means,
                       width,
                       label=model_name,
                       color=COLORS[model_name],
                       alpha=0.85,
                       yerr=stds,
                       capsize=5,
                       error_kw={'linewidth': 1.5,
                                'capthick': 1.5})

        # Highlight CausTab
        if model_name == 'CausTab':
            for bar in bars:
                bar.set_edgecolor('black')
                bar.set_linewidth(1.5)

        # Value labels — above error bar
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                   m + s + 0.012,
                   f'{m:.3f}',
                   ha='center', va='bottom',
                   fontsize=9, fontweight='bold',
                   color=COLORS[model_name])

    ax.set_xticks(x)
    ax.set_xticklabels(
        [regime_labels.get(r, r) for r in regimes],
        fontsize=10
    )
    ax.set_ylabel("Mean AUC-ROC", fontsize=11)
    ax.set_ylim(0.5, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle=':',
              linewidth=1, alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.9,
             fontsize=10)

    fig.tight_layout()
    save(fig, 'fig3_synthetic_regimes.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — NHANES temporal results
# ══════════════════════════════════════════════════════════════════════════════

def fig4_nhanes_temporal():
    """
    NHANES temporal forward-chaining results.
    Shows calibration advantage of CausTab.
    """
    csv = os.path.join(ROOT, 'experiments', 'results',
                       'temporal_split', 'all_temporal_results.csv')
    if not os.path.exists(csv):
        print("  SKIP fig4 — all_temporal_results.csv not found")
        return

    df = pd.read_csv(csv)

    splits = df['Split'].unique()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Figure 4. NHANES Temporal Forward-Chaining Evaluation\n"
        "Train on past cycles → Test on future cycles",
        fontweight='bold', y=1.01
    )

    metrics = [
        ('AUC',      'AUC-ROC',           axes[0,0]),
        ('Accuracy', 'Accuracy',           axes[0,1]),
        ('F1',       'F1 Score',           axes[1,0]),
        ('ECE',      'ECE (↓ = better)',   axes[1,1]),
    ]

    for metric, label, ax in metrics:
        for split_name in splits:
            sub       = df[df['Split'] == split_name]
            envs      = sorted(sub['Environment'].unique())
            short     = split_name.replace('Split_', 'Split ')

            for model_name in ['ERM', 'IRM', 'CausTab']:
                m_sub = sub[sub['Model'] == model_name]
                vals  = [m_sub[m_sub['Environment']==e
                               ][metric].values[0]
                        for e in envs if e in
                        m_sub['Environment'].values]
                e_labels = [e for e in envs
                           if e in m_sub['Environment'].values]

                linestyle = '-' if split_name == splits[0] else '--'
                ax.plot(e_labels, vals,
                       marker='o', markersize=6,
                       linestyle=linestyle,
                       color=COLORS[model_name],
                       alpha=0.9,
                       label=f"{model_name} ({short})"
                       if metric == 'AUC' else "_nolegend_")

        ax.set_xlabel("Test environment", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, pad=8)
        if metric == 'AUC':
            ax.legend(bbox_to_anchor=(1.01, 1),
                     loc='upper left',
                     fontsize=8,
                     framealpha=0.9)

    fig.tight_layout()
    save(fig, 'fig4_nhanes_temporal.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — NHANES calibration (ECE) — the cleanest win
# ══════════════════════════════════════════════════════════════════════════════

def fig5_calibration():
    """
    CausTab's clearest consistent advantage — calibration.
    Shows ECE across all NHANES experiments.
    """
    # Load standard split results
    std_csv = os.path.join(ROOT, 'experiments', 'results',
                          'evaluation_results.csv')
    tmp_csv = os.path.join(ROOT, 'experiments', 'results',
                          'temporal_split',
                          'all_temporal_results.csv')

    if not os.path.exists(std_csv):
        print("  SKIP fig5 — evaluation_results.csv not found")
        return

    std_df = pd.read_csv(std_csv)
    tmp_df = pd.read_csv(tmp_csv) if os.path.exists(tmp_csv) \
        else None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Figure 5. Calibration (ECE) Across NHANES Experiments\n"
        "CausTab consistently achieves best probability calibration",
        fontweight='bold', y=1.02
    )

    # Left: standard split ECE per environment
    ax   = axes[0]
    envs = ['2011-12', '2013-14', '2015-16', '2017-18']
    x    = np.arange(len(envs))
    w    = 0.25

    for i, model_name in enumerate(['ERM', 'IRM', 'CausTab']):
        sub  = std_df[std_df['Model'] == model_name]
        eces = [sub[sub['Environment']==e]['ECE'].values[0]
               if e in sub['Environment'].values else np.nan
               for e in envs]
        offset = (i - 1) * (w + 0.02)
        bars   = ax.bar(x + offset, eces, w,
                       label=model_name,
                       color=COLORS[model_name],
                       alpha=0.85)
        if model_name == 'CausTab':
            for bar in bars:
                bar.set_edgecolor('black')
                bar.set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=0)
    ax.set_xlabel("Survey cycle")
    ax.set_ylabel("ECE (lower = better calibrated)")
    ax.set_title("Standard split — ECE per cycle", pad=10)
    ax.legend(framealpha=0.9)

    # Right: mean ECE summary across all experiments
    ax2 = axes[1]

    experiment_labels = ['Standard\nsplit',
                        'Temporal\nSplit B',
                        'Temporal\nSplit C']
    result_files = [
        os.path.join(ROOT, 'experiments', 'results',
                    'evaluation_results.csv'),
        os.path.join(ROOT, 'experiments', 'results',
                    'temporal_split', 'split_b_results.csv'),
        os.path.join(ROOT, 'experiments', 'results',
                    'temporal_split', 'split_c_results.csv'),
    ]

    ece_data = {m: [] for m in ['ERM', 'IRM', 'CausTab']}

    for fpath in result_files:
        if not os.path.exists(fpath):
            for m in ece_data:
                ece_data[m].append(np.nan)
            continue
        sub = pd.read_csv(fpath)
        for m in ece_data:
            m_sub = sub[sub['Model'] == m] if 'Model' in sub.columns\
                else pd.DataFrame()
            if len(m_sub) > 0 and 'ECE' in m_sub.columns:
                ece_data[m].append(m_sub['ECE'].mean())
            else:
                ece_data[m].append(np.nan)

    x2 = np.arange(len(experiment_labels))
    w2 = 0.25

    for i, model_name in enumerate(['ERM', 'IRM', 'CausTab']):
        vals   = ece_data[model_name]
        offset = (i - 1) * (w2 + 0.02)
        bars   = ax2.bar(x2 + offset, vals, w2,
                        label=model_name,
                        color=COLORS[model_name],
                        alpha=0.85)
        if model_name == 'CausTab':
            for bar in bars:
                bar.set_edgecolor('black')
                bar.set_linewidth(1.5)

        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax2.text(bar.get_x() + bar.get_width()/2,
                        v + 0.001,
                        f'{v:.4f}',
                        ha='center', va='bottom',
                        fontsize=8)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(experiment_labels)
    ax2.set_ylabel("Mean ECE (lower = better)")
    ax2.set_title("Mean ECE across experiments\n"
                 "CausTab = best calibrated in all settings",
                 pad=10)
    ax2.legend(framealpha=0.9)

    fig.tight_layout()
    save(fig, 'fig5_calibration.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — SDI validation
# ══════════════════════════════════════════════════════════════════════════════

def fig6_sdi_validation():
    """
    Validates SDI as a regime predictor.
    Shows NHANES and UCI SDI in context.
    """
    csv = os.path.join(ROOT, 'experiments', 'results',
                       'synthetic', 'synthetic_results.csv')
    if not os.path.exists(csv):
        print("  SKIP fig6 — synthetic_results.csv not found")
        return

    df = pd.read_csv(csv)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        "Figure 6. Spurious Dominance Index (SDI) Validation\n"
        "Higher SDI predicts larger CausTab advantage over ERM",
        fontweight='bold', y=1.02
    )

    regimes     = df['Regime'].unique()
    sdi_vals    = []
    advantages  = []
    labels      = []
    colors_pts  = list(REGIME_COLORS.values())

    for regime in regimes:
        sub     = df[df['Regime'] == regime]
        sdi     = sub['SDI'].iloc[0]
        ct_auc  = sub[sub['Model']=='CausTab']['Mean_AUC'].values[0]
        erm_auc = sub[sub['Model']=='ERM']['Mean_AUC'].values[0]
        adv     = ct_auc - erm_auc
        sdi_vals.append(sdi)
        advantages.append(adv)
        short = regime.replace('Regime_', 'R')\
                      .replace('_Causal_Dominant', '1: Causal')\
                      .replace('_Mixed', '2: Mixed')\
                      .replace('_Spurious_Dominant', '3: Spurious')
        labels.append(short)

    # Add NHANES point (SDI computed during exploration)
    # NHANES SDI is approximately 1.7 from our earlier analysis
    nhanes_sdi = 1.7
    nhanes_adv = 0.0003  # from our NHANES results
    ax.scatter(nhanes_sdi, nhanes_adv,
              color='#8856A7', s=120,
              marker='D', zorder=5,
              label='NHANES (real data)')
    ax.annotate('NHANES',
               xy=(nhanes_sdi, nhanes_adv),
               xytext=(nhanes_sdi + 1.5, nhanes_adv + 0.002),
               fontsize=9,
               arrowprops=dict(arrowstyle='->',
                              color='gray',
                              lw=0.8))

    # Synthetic regime points
    for sdi, adv, label, color in zip(
            sdi_vals, advantages, labels, colors_pts):
        ax.scatter(sdi, adv, color=color,
                  s=150, zorder=5)
        offset_x = 1.5 if sdi < 20 else -3
        offset_y = 0.001
        ax.annotate(label,
                   xy=(sdi, adv),
                   xytext=(sdi + offset_x, adv + offset_y),
                   fontsize=9,
                   arrowprops=dict(arrowstyle='->',
                                  color='gray',
                                  lw=0.8)
                   if abs(offset_x) > 1 else None)

    ax.axhline(y=0, color='gray', linestyle='--',
              linewidth=1, alpha=0.7,
              label='No advantage (CausTab = ERM)')
    ax.set_xlabel("Spurious Dominance Index (SDI)", fontsize=11)
    ax.set_ylabel("CausTab AUC advantage over ERM", fontsize=11)
    ax.legend(framealpha=0.9, fontsize=9)

    fig.tight_layout()
    save(fig, 'fig6_sdi_validation.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Ablation study
# ══════════════════════════════════════════════════════════════════════════════

def fig7_ablation():
    """
    Clean ablation figure — one panel per regime plus NHANES.
    No overlapping text. Generous spacing.
    """
    syn_csv = os.path.join(ROOT, 'experiments', 'results',
                          'ablation', 'ablation_synthetic_results.csv')
    nha_csv = os.path.join(ROOT, 'experiments', 'results',
                          'ablation', 'ablation_nhanes_results.csv')

    if not os.path.exists(syn_csv):
        print("  SKIP fig7 — ablation results not found")
        return

    syn_df = pd.read_csv(syn_csv)
    nha_df = pd.read_csv(nha_csv) if os.path.exists(nha_csv) \
        else None

    variants = [
        'CausTab_Full',
        'CausTab_NoAnneal',
        'CausTab_NoWarmup',
        'CausTab_MeanPenalty',
        'CausTab_NoPenalty',
    ]
    var_labels = {
        'CausTab_Full':        'Full\n(proposed)',
        'CausTab_NoAnneal':    'No\nannealing',
        'CausTab_NoWarmup':    'No\nwarmup',
        'CausTab_MeanPenalty': 'Mean\npenalty',
        'CausTab_NoPenalty':   'No\npenalty',
    }
    var_colors = {
        'CausTab_Full':        COLORS['CausTab'],
        'CausTab_NoAnneal':    '#4C72B0',
        'CausTab_NoWarmup':    '#8172B2',
        'CausTab_MeanPenalty': '#CCB974',
        'CausTab_NoPenalty':   '#C44E52',
    }

    regimes = syn_df['Regime'].unique()
    n_cols  = len(regimes) + (1 if nha_df is not None else 0)

    fig, axes = plt.subplots(1, n_cols,
                            figsize=(4.5 * n_cols, 5.5),
                            sharey=False)
    if n_cols == 1:
        axes = [axes]

    fig.suptitle(
        "Figure 7. Ablation Study\n"
        "Each variant removes exactly one CausTab component",
        fontweight='bold', y=1.02
    )

    regime_titles = {
        'Regime_1_Causal_Dominant':
            'Synthetic — Regime 1\nCausal dominant',
        'Regime_2_Mixed':
            'Synthetic — Regime 2\nMixed',
        'Regime_3_Spurious_Dominant':
            'Synthetic — Regime 3\nSpurious dominant',
    }

    x      = np.arange(len(variants))
    width  = 0.6

    # Synthetic panels
    for ax, regime in zip(axes[:len(regimes)], regimes):
        sub    = syn_df[syn_df['Regime'] == regime]
        means  = [sub[sub['Variant']==v]['Mean_AUC'].values[0]
                 if len(sub[sub['Variant']==v]) > 0 else 0
                 for v in variants]
        stds   = [sub[sub['Variant']==v]['Std_AUC'].values[0]
                 if len(sub[sub['Variant']==v]) > 0 else 0
                 for v in variants]
        colors = [var_colors[v] for v in variants]

        bars = ax.bar(x, means,
                     width,
                     color=colors,
                     alpha=0.85,
                     yerr=stds,
                     capsize=5,
                     error_kw={'linewidth': 1.2})

        # Bold border on Full method
        bars[0].set_edgecolor('black')
        bars[0].set_linewidth(2)

        # Value labels — well above error bar
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                   m + s + 0.008,
                   f'{m:.3f}',
                   ha='center', va='bottom',
                   fontsize=8.5)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [var_labels[v] for v in variants],
            fontsize=9
        )
        y_min = max(0, min(means) - 0.06)
        y_max = max(means) + max(stds) + 0.05
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Mean AUC-ROC", fontsize=10)
        ax.set_title(regime_titles.get(regime, regime),
                    fontsize=11, pad=10)

    # NHANES panel
    if nha_df is not None and len(axes) > len(regimes):
        ax = axes[len(regimes)]
        means  = [nha_df[nha_df['Variant']==v]['Mean_AUC'].values[0]
                 if len(nha_df[nha_df['Variant']==v]) > 0 else 0
                 for v in variants]
        stds   = [nha_df[nha_df['Variant']==v]['Std_AUC'].values[0]
                 if len(nha_df[nha_df['Variant']==v]) > 0 else 0
                 for v in variants]
        colors = [var_colors[v] for v in variants]

        bars = ax.bar(x, means, width,
                     color=colors, alpha=0.85,
                     yerr=stds, capsize=5,
                     error_kw={'linewidth': 1.2})
        bars[0].set_edgecolor('black')
        bars[0].set_linewidth(2)

        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                   m + s + 0.001,
                   f'{m:.4f}',
                   ha='center', va='bottom',
                   fontsize=8.5)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [var_labels[v] for v in variants],
            fontsize=9
        )
        y_min = max(0, min(means) - 0.005)
        y_max = max(means) + max(stds) + 0.004
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Mean AUC-ROC", fontsize=10)
        ax.set_title("NHANES — Temporal Split B\nReal data validation",
                    fontsize=11, pad=10)

    fig.tight_layout(pad=2.0)
    save(fig, 'fig7_ablation.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — UCI Heart Disease results
# ══════════════════════════════════════════════════════════════════════════════

def fig8_uci_results():
    """
    UCI Heart Disease LOOCV results.
    Honest presentation including CausTab limitation.
    """
    csv = os.path.join(ROOT, 'experiments', 'results',
                       'uci_heart', 'uci_loocv_results.csv')
    if not os.path.exists(csv):
        print("  SKIP fig8 — uci_loocv_results.csv not found")
        return

    df = pd.read_csv(csv)

    hospitals = df['Test_Hospital'].unique()
    short_hospitals = [h.split('(')[0].strip() for h in hospitals]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Figure 8. UCI Heart Disease — "
        "Leave-One-Hospital-Out Evaluation\n"
        "Institutional distribution shift "
        "(four hospitals across three countries)",
        fontweight='bold', y=1.02
    )

    for ax, metric, label in zip(
        axes,
        ['AUC', 'F1', 'ECE'],
        ['AUC-ROC', 'F1 Score', 'ECE (↓ = better)']
    ):
        x     = np.arange(len(hospitals))
        width = 0.25

        for i, model_name in enumerate(['ERM', 'IRM', 'CausTab']):
            sub  = df[df['Model'] == model_name]
            vals = [sub[sub['Test_Hospital']==h
                       ][metric].values[0]
                   for h in hospitals]

            offset = (i - 1) * (width + 0.02)
            bars   = ax.bar(x + offset, vals, width,
                           label=model_name,
                           color=COLORS[model_name],
                           alpha=0.85)

            if model_name == 'CausTab':
                for bar in bars:
                    bar.set_edgecolor('black')
                    bar.set_linewidth(1.5)

            # Value labels — inside bar if tall enough
            for bar, v in zip(bars, vals):
                if bar.get_height() > 0.15:
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        bar.get_height() - 0.05,
                        f'{v:.2f}',
                        ha='center', va='top',
                        fontsize=8,
                        color='white',
                        fontweight='bold'
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(short_hospitals,
                          rotation=0, fontsize=9)
        ax.set_xlabel("Test hospital (held out)")
        ax.set_ylabel(label)
        ax.set_title(label, pad=10)
        ax.legend(framealpha=0.9, fontsize=9)
        ax.set_ylim(0, 1.05)

    fig.tight_layout(pad=2.0)
    save(fig, 'fig8_uci_results.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Summary comparison (the paper's concluding visual)
# ══════════════════════════════════════════════════════════════════════════════

def fig9_summary():
    """
    One clean summary figure showing the full story.
    Three experiments, three findings, one visual.
    """
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(
        "Figure 9. CausTab Summary — "
        "When Does Invariant Learning Help?\n"
        "Three experiments, three settings, consistent story",
        fontweight='bold', y=1.02
    )

    axes = fig.subplots(1, 3)

    # ── Panel 1: Synthetic — CausTab stable, IRM fails ─────────────────────
    ax = axes[0]
    syn_csv = os.path.join(ROOT, 'experiments', 'results',
                          'synthetic', 'synthetic_results.csv')
    if os.path.exists(syn_csv):
        syn = pd.read_csv(syn_csv)
        regimes    = syn['Regime'].unique()
        reg_labels = ['R1\nCausal', 'R2\nMixed', 'R3\nSpurious']
        x          = np.arange(len(regimes))
        w          = 0.25

        for i, m in enumerate(['ERM', 'IRM', 'CausTab']):
            sub  = syn[syn['Model'] == m]
            vals = [sub[sub['Regime']==r
                       ]['Mean_AUC'].values[0]
                   for r in regimes]
            stds = [sub[sub['Regime']==r
                       ]['Std_AUC'].values[0]
                   for r in regimes]
            offset = (i-1) * (w+0.02)
            ax.bar(x + offset, vals, w,
                  label=m,
                  color=COLORS[m],
                  alpha=0.85,
                  yerr=stds, capsize=4,
                  error_kw={'linewidth': 1.2})

        ax.set_xticks(x)
        ax.set_xticklabels(reg_labels)
        ax.set_ylabel("Mean AUC-ROC")
        ax.set_title("Synthetic data\n"
                    "CausTab ≥ ERM always\n"
                    "IRM fails in spurious regime",
                    fontsize=10, pad=8)
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=9, framealpha=0.9)

    # ── Panel 2: NHANES — calibration advantage ─────────────────────────
    ax2 = axes[1]
    std_csv = os.path.join(ROOT, 'experiments', 'results',
                          'evaluation_results.csv')
    if os.path.exists(std_csv):
        std = pd.read_csv(std_csv)
        models     = ['ERM', 'IRM', 'CausTab']
        mean_aucs  = [std[std['Model']==m]['AUC'].mean()
                     for m in models]
        mean_eces  = [std[std['Model']==m]['ECE'].mean()
                     for m in models]
        x2         = np.arange(len(models))
        colors2    = [COLORS[m] for m in models]

        bars = ax2.bar(x2, mean_aucs,
                      color=colors2, alpha=0.85,
                      width=0.5)
        bars[2].set_edgecolor('black')
        bars[2].set_linewidth(2)

        # Add ECE as text annotation
        for bar, auc, ece, m in zip(
                bars, mean_aucs, mean_eces, models):
            ax2.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.002,
                    f'AUC={auc:.3f}\nECE={ece:.3f}',
                    ha='center', va='bottom',
                    fontsize=8.5)

        ax2.set_xticks(x2)
        ax2.set_xticklabels(models)
        ax2.set_ylabel("Mean AUC-ROC")
        ax2.set_title("NHANES real data\n"
                     "CausTab = best calibration\n"
                     "(causal-dominant regime)",
                     fontsize=10, pad=8)
        ax2.set_ylim(0.7, 0.9)

    # ── Panel 3: UCI — boundary condition ─────────────────────────────
    ax3 = axes[2]
    uci_csv = os.path.join(ROOT, 'experiments', 'results',
                          'uci_heart', 'uci_summary.csv')
    if os.path.exists(uci_csv):
        uci    = pd.read_csv(uci_csv)
        models = ['ERM', 'IRM', 'CausTab']
        aucs   = [uci[uci['Model']==m]['Mean_AUC'].values[0]
                 for m in models]
        eces   = [uci[uci['Model']==m]['Mean_ECE'].values[0]
                 for m in models]
        x3     = np.arange(len(models))
        colors3 = [COLORS[m] for m in models]

        bars = ax3.bar(x3, aucs,
                      color=colors3, alpha=0.85,
                      width=0.5)
        bars[2].set_edgecolor('black')
        bars[2].set_linewidth(2)

        for bar, auc, ece, m in zip(
                bars, aucs, eces, models):
            ax3.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.004,
                    f'AUC={auc:.3f}',
                    ha='center', va='bottom',
                    fontsize=8.5)

        ax3.set_xticks(x3)
        ax3.set_xticklabels(models)
        ax3.set_ylabel("Mean AUC-ROC")
        ax3.set_title("UCI Heart Disease\n"
                     "Boundary condition:\n"
                     "extreme prevalence shift",
                     fontsize=10, pad=8)
        ax3.set_ylim(0.4, 0.9)

        # Add annotation explaining limitation
        ax3.text(0.5, 0.08,
                "Note: CausTab underperforms when\n"
                "base rates differ across environments",
                transform=ax3.transAxes,
                ha='center', fontsize=8,
                style='italic',
                bbox=dict(boxstyle='round,pad=0.4',
                         facecolor='#FFF9C4',
                         edgecolor='#F5A623',
                         alpha=0.9))

    fig.tight_layout(pad=2.5)
    save(fig, 'fig9_summary.png')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("CausTab — Publication Quality Plot Generator")
    print(f"Output: experiments/plots/publication/")
    print("="*60)

    print("\nGenerating figures...")
    fig1_nhanes_shift_evidence()
    fig2_irm_failure()
    fig3_synthetic_regimes()
    fig4_nhanes_temporal()
    fig5_calibration()
    fig6_sdi_validation()
    fig7_ablation()
    fig8_uci_results()
    fig9_summary()

    print(f"\n{'='*60}")
    print("ALL FIGURES COMPLETE")
    print(f"Saved to: experiments/plots/publication/")
    print(f"{'='*60}")