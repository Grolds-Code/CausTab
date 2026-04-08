"""
CausTab — Publication Quality Plots — Style B
NeurIPS/ICML professional style

Design principles Style B:
    - Subtle grid lines (evidence of rigor)
    - Stronger colors with higher contrast
    - Thicker lines and markers
    - Shaded confidence regions
    - Professional gray backgrounds on figure titles
    - Consistent serif-free typography
    - Bold axis labels
    - Tighter but clean spacing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import os
import sys

# ── Style B Global Settings ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':          'DejaVu Sans',
    'font.size':            11,
    'axes.titlesize':       12,
    'axes.titleweight':     'bold',
    'axes.labelsize':       11,
    'axes.labelweight':     'bold',
    'xtick.labelsize':      10,
    'ytick.labelsize':      10,
    'legend.fontsize':      9,
    'legend.framealpha':    0.92,
    'legend.edgecolor':     '#cccccc',
    'figure.titlesize':     13,
    'figure.titleweight':   'bold',
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'axes.grid':            True,
    'grid.alpha':           0.4,
    'grid.linewidth':       0.6,
    'grid.color':           '#b0b0b0',
    'axes.axisbelow':       True,
    'lines.linewidth':      2.5,
    'lines.markersize':     8,
    'patch.linewidth':      1.0,
    'figure.dpi':           150,
    'savefig.dpi':          300,
    'savefig.bbox':         'tight',
    'savefig.pad_inches':   0.2,
    'axes.facecolor':       '#f8f8f8',
    'figure.facecolor':     'white',
    'axes.edgecolor':       '#888888',
    'xtick.color':          '#444444',
    'ytick.color':          '#444444',
    'axes.labelcolor':      '#222222',
    'text.color':           '#222222',
})

# ── Color palette Style B — stronger, higher contrast ─────────────────────────
COLORS = {
    'ERM':     '#1565C0',   # deep blue
    'IRM':     '#B71C1C',   # deep red
    'CausTab': '#1B5E20',   # deep green
}

FILL_COLORS = {
    'ERM':     '#90CAF9',
    'IRM':     '#EF9A9A',
    'CausTab': '#A5D6A7',
}

REGIME_COLORS = {
    'Regime_1_Causal_Dominant':   '#2E7D32',
    'Regime_2_Mixed':             '#E65100',
    'Regime_3_Spurious_Dominant': '#6A1B9A',
}

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_ROOT = os.path.join(ROOT, 'experiments', 'plots', 'publicationB')
os.makedirs(PLOT_ROOT, exist_ok=True)

def save(fig, name):
    path = os.path.join(PLOT_ROOT, name)
    fig.savefig(path, facecolor='white')
    plt.close(fig)
    print(f"  Saved: {name}")


def style_ax(ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent Style B formatting to an axis."""
    if title:
        ax.set_title(title, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(length=4, width=1.0)


def add_value_labels(ax, bars, values, stds=None,
                     fmt='.3f', fontsize=9, color=None):
    """
    Add value labels above bars with safe spacing.
    Never overlaps with error bars.
    """
    for bar, v in zip(bars, values):
        offset = 0.0
        if stds is not None:
            idx    = list(values).index(v)
            offset = stds[idx] + 0.008
        else:
            offset = bar.get_height() * 0.015 + 0.005
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f'{v:{fmt}}',
            ha='center', va='bottom',
            fontsize=fontsize,
            color=color or '#333333',
            fontweight='bold'
        )


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — NHANES shift evidence
# ══════════════════════════════════════════════════════════════════════════════

def fig1_nhanes_shift_evidence():
    csv = os.path.join(ROOT, 'data', 'tables',
                       'table3_correlation_shift.csv')
    if not os.path.exists(csv):
        print("  SKIP fig1")
        return

    df = pd.read_csv(csv)
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
    df['Feature'] = df['Feature'].map(lambda x: name_map.get(x, x))
    cycles        = ['2011-12', '2013-14', '2015-16', '2017-18']
    available     = [c for c in cycles if c in df.columns]
    df            = df.sort_values('Range', ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Distribution Shift Evidence in NHANES (2011–2018)\n"
        "Feature–outcome correlations vary across survey cycles",
        y=1.01
    )

    # Heatmap
    ax          = axes[0]
    corr_matrix = df.set_index('Feature')[available].values
    features    = df['Feature'].tolist()

    im = ax.imshow(corr_matrix, cmap='RdBu_r',
                  vmin=-0.5, vmax=0.5, aspect='auto')
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels(available)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_title("Pearson correlation with hypertension\nper NHANES cycle")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation coefficient', fontsize=10)

    for i in range(len(features)):
        for j in range(len(available)):
            val = corr_matrix[i, j]
            ax.text(j, i, f'{val:.2f}',
                   ha='center', va='center',
                   fontsize=8.5,
                   color='white' if abs(val) > 0.28
                   else '#333333')

    ax.grid(False)
    style_ax(ax)

    # Range bars
    ax2    = axes[1]
    ranges = df['Range'].values
    colors_bar = ['#B71C1C' if r > 0.02 else '#1B5E20'
                 for r in ranges]
    ax2.barh(features, ranges,
            color=colors_bar, alpha=0.88,
            height=0.65, edgecolor='white',
            linewidth=0.5)
    ax2.axvline(x=0.02, color='#555555',
               linestyle='--', linewidth=1.5,
               alpha=0.8, label='Threshold (0.02)')

    stable_p   = mpatches.Patch(
        color='#1B5E20', alpha=0.88,
        label='Stable — causal signal')
    unstable_p = mpatches.Patch(
        color='#B71C1C', alpha=0.88,
        label='Unstable — spurious/mixed')
    ax2.legend(handles=[stable_p, unstable_p],
              loc='lower right', fontsize=9)

    style_ax(ax2,
            title='Feature stability across NHANES cycles',
            xlabel='Correlation range  (max − min across cycles)')

    fig.tight_layout(pad=1.5)
    save(fig, 'fig1_nhanes_shift_evidence.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — IRM failure (centerpiece)
# ══════════════════════════════════════════════════════════════════════════════

def fig2_irm_failure():
    csv = os.path.join(ROOT, 'experiments', 'results',
                       'irm_analysis', 'spurious_sweep_results.csv')
    if not os.path.exists(csv):
        print("  SKIP fig2")
        return

    df        = pd.read_csv(csv)
    strengths = sorted(df['Spurious_strength'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "IRM Failure on Tabular Data\n"
        "Performance vs spurious feature strength "
        "(complete correlation collapse at test time)",
        y=1.01
    )

    for ax, metric, ylabel, title, legend_loc in zip(
        axes,
        ['Mean_AUC', 'Std_AUC'],
        ['Mean AUC-ROC (± std across seeds)',
         'Std of AUC across seeds  (↓ = more stable)'],
        ['Predictive performance',
         'Stability (reliability)'],
        ['lower left', 'upper left']
    ):
        for model_name in ['ERM', 'IRM', 'CausTab']:
            sub  = df[df['Model'] == model_name]
            vals = [sub[sub['Spurious_strength']==s
                       ][metric].values[0]
                   for s in strengths]

            ax.plot(strengths, vals,
                   marker='o',
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
                    alpha=0.15,
                    color=FILL_COLORS[model_name]
                )

        if metric == 'Mean_AUC':
            ax.axhline(y=0.5, color='#888888',
                      linestyle=':', linewidth=1.5,
                      alpha=0.8, label='Random (AUC=0.5)')
            ax.set_ylim(0.45, 1.02)

        style_ax(ax,
                title=title,
                xlabel='Spurious feature strength in training',
                ylabel=ylabel)
        ax.legend(loc=legend_loc)

        # Annotate worst IRM point
        if metric == 'Mean_AUC':
            irm_sub  = df[df['Model']=='IRM']
            erm_sub  = df[df['Model']=='ERM']
            gaps     = []
            for s in strengths:
                irm_v = irm_sub[irm_sub['Spurious_strength']==s
                                ]['Mean_AUC'].values[0]
                erm_v = erm_sub[erm_sub['Spurious_strength']==s
                                ]['Mean_AUC'].values[0]
                gaps.append(irm_v - erm_v)
            worst_idx = np.argmin(gaps)
            worst_s   = strengths[worst_idx]
            worst_gap = gaps[worst_idx]
            ax.annotate(
                f'Max IRM gap:\n{worst_gap:+.3f} AUC',
                xy=(worst_s,
                    irm_sub[irm_sub['Spurious_strength']==worst_s
                            ]['Mean_AUC'].values[0]),
                xytext=(worst_s + 0.3,
                        irm_sub[irm_sub['Spurious_strength']==worst_s
                                ]['Mean_AUC'].values[0] + 0.08),
                fontsize=9,
                color=COLORS['IRM'],
                arrowprops=dict(arrowstyle='->',
                               color=COLORS['IRM'],
                               lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3',
                         facecolor='white',
                         edgecolor=COLORS['IRM'],
                         alpha=0.9)
            )

    fig.tight_layout(pad=1.5)
    save(fig, 'fig2_irm_failure.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Synthetic regimes
# ══════════════════════════════════════════════════════════════════════════════

def fig3_synthetic_regimes():
    csv = os.path.join(ROOT, 'experiments', 'results',
                       'synthetic', 'synthetic_results.csv')
    if not os.path.exists(csv):
        print("  SKIP fig3")
        return

    df      = pd.read_csv(csv)
    regimes = list(df['Regime'].unique())

    regime_labels = {
        'Regime_1_Causal_Dominant':
            'Regime 1\nCausal dominant\nSDI = 1.67',
        'Regime_2_Mixed':
            'Regime 2\nMixed\nSDI = 9.62',
        'Regime_3_Spurious_Dominant':
            'Regime 3\nSpurious dominant\nSDI = 48.08',
    }

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.suptitle(
        "Synthetic Experiment — Performance Across "
        "Three Spurious-Correlation Regimes\n"
        "Mean AUC-ROC ± std across 5 random seeds",
        y=1.01
    )

    x     = np.arange(len(regimes))
    width = 0.22

    for i, model_name in enumerate(['ERM', 'IRM', 'CausTab']):
        sub    = df[df['Model'] == model_name]
        means  = [sub[sub['Regime']==r]['Mean_AUC'].values[0]
                 for r in regimes]
        stds   = [sub[sub['Regime']==r]['Std_AUC'].values[0]
                 for r in regimes]
        offset = (i - 1) * (width + 0.03)

        bars = ax.bar(x + offset, means, width,
                     label=model_name,
                     color=COLORS[model_name],
                     alpha=0.88,
                     yerr=stds,
                     capsize=5,
                     error_kw={'linewidth': 1.8,
                              'capthick': 1.8,
                              'ecolor': COLORS[model_name]},
                     edgecolor='white',
                     linewidth=0.5)

        if model_name == 'CausTab':
            for bar in bars:
                bar.set_edgecolor('black')
                bar.set_linewidth(2)

        add_value_labels(ax, bars, means, stds,
                        fmt='.3f', fontsize=8.5,
                        color=COLORS[model_name])

    ax.set_xticks(x)
    ax.set_xticklabels(
        [regime_labels.get(r, r) for r in regimes],
        fontsize=10
    )
    ax.set_ylim(0.5, 1.08)
    ax.axhline(y=0.5, color='#888888',
              linestyle=':', linewidth=1.2, alpha=0.6)
    ax.legend(loc='upper right', fontsize=10)
    style_ax(ax, ylabel='Mean AUC-ROC')

    # Add regime background shading
    for i, regime in enumerate(regimes):
        color = list(REGIME_COLORS.values())[i]
        ax.axvspan(i - 0.45, i + 0.45,
                  alpha=0.04, color=color, zorder=0)

    fig.tight_layout(pad=1.5)
    save(fig, 'fig3_synthetic_regimes.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — NHANES temporal results
# ══════════════════════════════════════════════════════════════════════════════

def fig4_nhanes_temporal():
    csv = os.path.join(ROOT, 'experiments', 'results',
                       'temporal_split', 'all_temporal_results.csv')
    if not os.path.exists(csv):
        print("  SKIP fig4")
        return

    df     = pd.read_csv(csv)
    splits = df['Split'].unique()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "NHANES Temporal Forward-Chaining Evaluation\n"
        "Train on past survey cycles → Test on future cycles",
        y=1.01
    )

    linestyles = {splits[0]: '-', splits[1]: '--'}
    markers    = {splits[0]: 'o', splits[1]: 's'}

    for ax, metric, label in zip(
        axes,
        ['AUC', 'ECE'],
        ['AUC-ROC', 'ECE (↓ = better calibrated)']
    ):
        for split_name in splits:
            sub   = df[df['Split'] == split_name]
            envs  = sorted(sub['Environment'].unique())
            short = split_name.replace('Split_', 'Split ')

            for model_name in ['ERM', 'IRM', 'CausTab']:
                m_sub = sub[sub['Model'] == model_name]
                vals  = [m_sub[m_sub['Environment']==e
                               ][metric].values[0]
                        for e in envs
                        if e in m_sub['Environment'].values]
                e_labs = [e for e in envs
                         if e in m_sub['Environment'].values]

                ax.plot(e_labs, vals,
                       marker=markers[split_name],
                       markersize=7,
                       linestyle=linestyles[split_name],
                       color=COLORS[model_name],
                       alpha=0.9,
                       label=f"{model_name} ({short})")

                # Fill CI band for AUC if CI columns exist
                if metric == 'AUC' and 'AUC_CI_lo' in m_sub.columns:
                    lo = [m_sub[m_sub['Environment']==e
                               ]['AUC_CI_lo'].values[0]
                         for e in e_labs]
                    hi = [m_sub[m_sub['Environment']==e
                               ]['AUC_CI_hi'].values[0]
                         for e in e_labs]
                    ax.fill_between(e_labs, lo, hi,
                                   alpha=0.08,
                                   color=FILL_COLORS[model_name])

        style_ax(ax,
                title=label,
                xlabel='Test environment (survey cycle)',
                ylabel=label)
        ax.legend(bbox_to_anchor=(1.01, 1),
                 loc='upper left',
                 fontsize=8,
                 framealpha=0.92)

    fig.tight_layout(pad=1.5)
    save(fig, 'fig4_nhanes_temporal.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Calibration (ECE) — CausTab's clearest win
# ══════════════════════════════════════════════════════════════════════════════

def fig5_calibration():
    std_csv = os.path.join(ROOT, 'experiments', 'results',
                          'evaluation_results.csv')
    if not os.path.exists(std_csv):
        print("  SKIP fig5")
        return

    std_df = pd.read_csv(std_csv)

    result_files = {
        'Standard split': std_csv,
        'Temporal Split B': os.path.join(
            ROOT, 'experiments', 'results',
            'temporal_split', 'split_b_results.csv'),
        'Temporal Split C': os.path.join(
            ROOT, 'experiments', 'results',
            'temporal_split', 'split_c_results.csv'),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Probability Calibration (ECE) Across All NHANES Experiments\n"
        "CausTab consistently achieves the lowest calibration error",
        y=1.01
    )

    # Left: ECE per environment (standard split)
    ax   = axes[0]
    envs = ['2011-12', '2013-14', '2015-16', '2017-18']
    x    = np.arange(len(envs))
    w    = 0.25

    for i, model_name in enumerate(['ERM', 'IRM', 'CausTab']):
        sub  = std_df[std_df['Model'] == model_name]
        eces = [sub[sub['Environment']==e]['ECE'].values[0]
               if e in sub['Environment'].values else np.nan
               for e in envs]
        offset = (i-1) * (w+0.02)
        bars   = ax.bar(x + offset, eces, w,
                       label=model_name,
                       color=COLORS[model_name],
                       alpha=0.88,
                       edgecolor='white',
                       linewidth=0.5)
        if model_name == 'CausTab':
            for bar in bars:
                bar.set_edgecolor('black')
                bar.set_linewidth(1.8)

        add_value_labels(ax, bars, eces, fmt='.3f',
                        fontsize=8, color=COLORS[model_name])

    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.legend(fontsize=9)
    style_ax(ax,
            title='ECE per cycle — standard split',
            xlabel='Survey cycle',
            ylabel='ECE (lower = better calibrated)')

    # Right: mean ECE across all experiments
    ax2         = axes[1]
    exp_labels  = list(result_files.keys())
    ece_data    = {m: [] for m in ['ERM', 'IRM', 'CausTab']}

    for fpath in result_files.values():
        if not os.path.exists(fpath):
            for m in ece_data:
                ece_data[m].append(np.nan)
            continue
        sub = pd.read_csv(fpath)
        for m in ece_data:
            m_sub = sub[sub['Model']==m] \
                if 'Model' in sub.columns else pd.DataFrame()
            if len(m_sub) > 0 and 'ECE' in m_sub.columns:
                ece_data[m].append(m_sub['ECE'].mean())
            else:
                ece_data[m].append(np.nan)

    x2 = np.arange(len(exp_labels))
    w2 = 0.25

    for i, model_name in enumerate(['ERM', 'IRM', 'CausTab']):
        vals   = ece_data[model_name]
        offset = (i-1) * (w2+0.02)
        bars   = ax2.bar(x2 + offset, vals, w2,
                        label=model_name,
                        color=COLORS[model_name],
                        alpha=0.88,
                        edgecolor='white',
                        linewidth=0.5)
        if model_name == 'CausTab':
            for bar in bars:
                bar.set_edgecolor('black')
                bar.set_linewidth(1.8)

        add_value_labels(ax2, bars, vals, fmt='.4f',
                        fontsize=8,
                        color=COLORS[model_name])

    ax2.set_xticks(x2)
    ax2.set_xticklabels(exp_labels, fontsize=9)
    ax2.legend(fontsize=9)
    style_ax(ax2,
            title='Mean ECE — all NHANES experiments\n'
                  'CausTab = best in every setting',
            xlabel='Experiment',
            ylabel='Mean ECE (lower = better)')

    fig.tight_layout(pad=1.5)
    save(fig, 'fig5_calibration.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — SDI validation
# ══════════════════════════════════════════════════════════════════════════════

def fig6_sdi_validation():
    csv = os.path.join(ROOT, 'experiments', 'results',
                       'synthetic', 'synthetic_results.csv')
    if not os.path.exists(csv):
        print("  SKIP fig6")
        return

    df = pd.read_csv(csv)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.suptitle(
        "Spurious Dominance Index (SDI) Validation\n"
        "SDI predicts CausTab advantage — higher SDI → larger benefit",
        y=1.01
    )

    regimes    = df['Regime'].unique()
    sdi_vals   = []
    advantages = []
    labels     = []
    colors_pts = list(REGIME_COLORS.values())

    for regime in regimes:
        sub     = df[df['Regime'] == regime]
        sdi     = sub['SDI'].iloc[0]
        ct_auc  = sub[sub['Model']=='CausTab']['Mean_AUC'].values[0]
        erm_auc = sub[sub['Model']=='ERM']['Mean_AUC'].values[0]
        adv     = ct_auc - erm_auc
        ct_std  = sub[sub['Model']=='CausTab']['Std_AUC'].values[0]
        sdi_vals.append(sdi)
        advantages.append(adv)
        short = (regime
                 .replace('Regime_1_Causal_Dominant', 'R1: Causal')
                 .replace('Regime_2_Mixed',           'R2: Mixed')
                 .replace('Regime_3_Spurious_Dominant','R3: Spurious'))
        labels.append(short)

    # NHANES point
    nhanes_sdi = 1.7
    nhanes_adv = 0.0003
    ax.scatter(nhanes_sdi, nhanes_adv,
              color='#6A1B9A', s=140,
              marker='D', zorder=5,
              edgecolors='black', linewidths=1.2,
              label='NHANES (real — temporal shift)')
    ax.annotate(
        'NHANES\n(temporal)',
        xy=(nhanes_sdi, nhanes_adv),
        xytext=(nhanes_sdi + 3, nhanes_adv + 0.003),
        fontsize=9, color='#6A1B9A',
        arrowprops=dict(arrowstyle='->',
                       color='#6A1B9A', lw=1.2)
    )

    for sdi, adv, label, color in zip(
            sdi_vals, advantages, labels, colors_pts):
        ax.scatter(sdi, adv, color=color,
                  s=160, zorder=5,
                  edgecolors='black', linewidths=1.2)
        offset_x = 2 if sdi < 25 else -8
        ax.annotate(
            label,
            xy=(sdi, adv),
            xytext=(sdi + offset_x, adv + 0.0015),
            fontsize=9, color=color,
            fontweight='bold',
            arrowprops=dict(arrowstyle='->',
                           color=color, lw=1.0)
        )

    ax.axhline(y=0, color='#555555',
              linestyle='--', linewidth=1.5,
              alpha=0.7,
              label='No advantage (CausTab = ERM)')
    ax.axhspan(-0.002, 0.002, alpha=0.08,
              color='gray', label='±0.002 tolerance band')

    style_ax(ax,
            xlabel='Spurious Dominance Index (SDI)',
            ylabel='CausTab AUC advantage over ERM')
    ax.legend(loc='upper left', fontsize=9)

    fig.tight_layout(pad=1.5)
    save(fig, 'fig6_sdi_validation.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Ablation study
# ══════════════════════════════════════════════════════════════════════════════

def fig7_ablation():
    syn_csv = os.path.join(ROOT, 'experiments', 'results',
                          'ablation', 'ablation_synthetic_results.csv')
    nha_csv = os.path.join(ROOT, 'experiments', 'results',
                          'ablation', 'ablation_nhanes_results.csv')

    if not os.path.exists(syn_csv):
        print("  SKIP fig7")
        return

    syn_df = pd.read_csv(syn_csv)
    nha_df = pd.read_csv(nha_csv) if os.path.exists(nha_csv) \
        else None

    variants = [
        'CausTab_Full', 'CausTab_NoAnneal',
        'CausTab_NoWarmup', 'CausTab_MeanPenalty',
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
        'CausTab_NoAnneal':    '#1565C0',
        'CausTab_NoWarmup':    '#6A1B9A',
        'CausTab_MeanPenalty': '#E65100',
        'CausTab_NoPenalty':   '#B71C1C',
    }

    regimes = list(syn_df['Regime'].unique())
    n_cols  = len(regimes) + (1 if nha_df is not None else 0)

    regime_titles = {
        'Regime_1_Causal_Dominant':
            'Synthetic — Regime 1\nCausal dominant',
        'Regime_2_Mixed':
            'Synthetic — Regime 2\nMixed',
        'Regime_3_Spurious_Dominant':
            'Synthetic — Regime 3\nSpurious dominant',
    }

    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(4.8 * n_cols, 5.5),
        sharey=False
    )
    if n_cols == 1:
        axes = [axes]

    fig.suptitle(
        "Ablation Study — Component Justification\n"
        "Each variant removes exactly one CausTab component",
        y=1.01
    )

    x     = np.arange(len(variants))
    width = 0.62

    for ax, regime in zip(axes[:len(regimes)], regimes):
        sub    = syn_df[syn_df['Regime'] == regime]
        means  = [sub[sub['Variant']==v]['Mean_AUC'].values[0]
                 if len(sub[sub['Variant']==v]) > 0 else 0
                 for v in variants]
        stds   = [sub[sub['Variant']==v]['Std_AUC'].values[0]
                 if len(sub[sub['Variant']==v]) > 0 else 0
                 for v in variants]
        colors = [var_colors[v] for v in variants]

        bars = ax.bar(x, means, width,
                     color=colors, alpha=0.88,
                     yerr=stds, capsize=5,
                     error_kw={'linewidth': 1.8,
                              'capthick': 1.8},
                     edgecolor='white', linewidth=0.5)
        bars[0].set_edgecolor('black')
        bars[0].set_linewidth(2.5)

        add_value_labels(ax, bars, means, stds,
                        fmt='.3f', fontsize=8.5)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [var_labels[v] for v in variants],
            fontsize=9
        )
        y_min = max(0.0, min(means) - 0.06)
        y_max = max(means) + max(stds) + 0.06
        ax.set_ylim(y_min, y_max)
        style_ax(ax,
                title=regime_titles.get(regime, regime),
                ylabel='Mean AUC-ROC')

    if nha_df is not None and len(axes) > len(regimes):
        ax     = axes[len(regimes)]
        means  = [nha_df[nha_df['Variant']==v]['Mean_AUC'].values[0]
                 if len(nha_df[nha_df['Variant']==v]) > 0 else 0
                 for v in variants]
        stds   = [nha_df[nha_df['Variant']==v]['Std_AUC'].values[0]
                 if len(nha_df[nha_df['Variant']==v]) > 0 else 0
                 for v in variants]
        colors = [var_colors[v] for v in variants]

        bars = ax.bar(x, means, width,
                     color=colors, alpha=0.88,
                     yerr=stds, capsize=5,
                     error_kw={'linewidth': 1.8,
                              'capthick': 1.8},
                     edgecolor='white', linewidth=0.5)
        bars[0].set_edgecolor('black')
        bars[0].set_linewidth(2.5)

        add_value_labels(ax, bars, means, stds,
                        fmt='.4f', fontsize=8.5)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [var_labels[v] for v in variants],
            fontsize=9
        )
        y_min = max(0.0, min(means) - 0.005)
        y_max = max(means) + max(stds) + 0.005
        ax.set_ylim(y_min, y_max)
        style_ax(ax,
                title='NHANES — Temporal Split B\nReal data validation',
                ylabel='Mean AUC-ROC')

    fig.tight_layout(pad=2.5)
    save(fig, 'fig7_ablation.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — UCI Heart Disease
# ══════════════════════════════════════════════════════════════════════════════

def fig8_uci_results():
    csv = os.path.join(ROOT, 'experiments', 'results',
                       'uci_heart', 'uci_loocv_results.csv')
    if not os.path.exists(csv):
        print("  SKIP fig8")
        return

    df        = pd.read_csv(csv)
    hospitals = list(df['Test_Hospital'].unique())
    short_h   = [h.split('(')[0].strip() for h in hospitals]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    fig.suptitle(
        "UCI Heart Disease — Leave-One-Hospital-Out Evaluation\n"
        "Institutional distribution shift "
        "(4 hospitals across 3 countries)",
        y=1.01
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
            errors = None
            if 'AUC_CI_lo' in sub.columns and metric == 'AUC':
                lo     = [sub[sub['Test_Hospital']==h
                             ]['AUC_CI_lo'].values[0]
                         for h in hospitals]
                hi     = [sub[sub['Test_Hospital']==h
                             ]['AUC_CI_hi'].values[0]
                         for h in hospitals]
                errors = [
                    [v-l for v,l in zip(vals,lo)],
                    [h-v for v,h in zip(vals,hi)]
                ]

            offset = (i-1) * (width+0.02)
            kwargs = dict(
                label   = model_name,
                color   = COLORS[model_name],
                alpha   = 0.88,
                edgecolor='white',
                linewidth=0.5
            )
            if errors:
                kwargs.update(dict(
                    yerr=errors, capsize=4,
                    error_kw={'linewidth':1.5,
                             'capthick':1.5,
                             'ecolor':COLORS[model_name]}
                ))
            bars = ax.bar(x + offset, vals, width, **kwargs)

            if model_name == 'CausTab':
                for bar in bars:
                    bar.set_edgecolor('black')
                    bar.set_linewidth(1.8)

            # Compact value labels inside bar
            for bar, v in zip(bars, vals):
                h = bar.get_height()
                if h > 0.12:
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        h/2,
                        f'{v:.2f}',
                        ha='center', va='center',
                        fontsize=8,
                        color='white',
                        fontweight='bold'
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(short_h, rotation=0, fontsize=9)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        style_ax(ax,
                title=label,
                xlabel='Test hospital (held out)',
                ylabel=label)

    # Add boundary condition note on last panel
    axes[2].text(
        0.5, -0.22,
        "Note: CausTab underperforms ERM when hospital base rates\n"
        "differ substantially — boundary condition identified.",
        transform=axes[2].transAxes,
        ha='center', fontsize=8.5,
        style='italic', color='#555555',
        bbox=dict(boxstyle='round,pad=0.4',
                 facecolor='#FFF9C4',
                 edgecolor='#F5A623',
                 alpha=0.95)
    )

    fig.tight_layout(pad=2.0)
    save(fig, 'fig8_uci_results.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Summary (the paper's concluding visual)
# ══════════════════════════════════════════════════════════════════════════════

def fig9_summary():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(
        "CausTab — When Does Invariant Learning Help?\n"
        "Three experiments, three settings, consistent story",
        y=1.01
    )

    # ── Panel 1: Synthetic ─────────────────────────────────────────────────
    ax      = axes[0]
    syn_csv = os.path.join(ROOT, 'experiments', 'results',
                          'synthetic', 'synthetic_results.csv')
    if os.path.exists(syn_csv):
        syn     = pd.read_csv(syn_csv)
        regimes = list(syn['Regime'].unique())
        x       = np.arange(len(regimes))
        w       = 0.22
        r_labs  = ['R1\nCausal', 'R2\nMixed', 'R3\nSpurious']

        for i, m in enumerate(['ERM', 'IRM', 'CausTab']):
            sub    = syn[syn['Model']==m]
            means  = [sub[sub['Regime']==r
                         ]['Mean_AUC'].values[0]
                     for r in regimes]
            stds   = [sub[sub['Regime']==r
                         ]['Std_AUC'].values[0]
                     for r in regimes]
            offset = (i-1) * (w+0.03)
            bars   = ax.bar(x+offset, means, w,
                           label=m,
                           color=COLORS[m],
                           alpha=0.88,
                           yerr=stds, capsize=4,
                           error_kw={'linewidth':1.5,
                                    'ecolor':COLORS[m]},
                           edgecolor='white')
            if m == 'CausTab':
                for bar in bars:
                    bar.set_edgecolor('black')
                    bar.set_linewidth(2)

        ax.set_xticks(x)
        ax.set_xticklabels(r_labs, fontsize=10)
        ax.set_ylim(0.5, 1.1)
        ax.legend(fontsize=9)
        style_ax(ax,
                title='Synthetic data\nCausTab ≥ ERM always\n'
                      'IRM fails in R3',
                ylabel='Mean AUC-ROC')

    # ── Panel 2: NHANES ────────────────────────────────────────────────────
    ax2     = axes[1]
    std_csv = os.path.join(ROOT, 'experiments', 'results',
                          'evaluation_results.csv')
    if os.path.exists(std_csv):
        std    = pd.read_csv(std_csv)
        models = ['ERM', 'IRM', 'CausTab']
        aucs   = [std[std['Model']==m]['AUC'].mean()
                 for m in models]
        eces   = [std[std['Model']==m]['ECE'].mean()
                 for m in models]
        x2     = np.arange(len(models))
        colors2 = [COLORS[m] for m in models]

        bars = ax2.bar(x2, aucs, 0.5,
                      color=colors2, alpha=0.88,
                      edgecolor='white')
        bars[2].set_edgecolor('black')
        bars[2].set_linewidth(2)

        for bar, auc, ece in zip(bars, aucs, eces):
            ax2.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f'AUC={auc:.3f}\nECE={ece:.4f}',
                ha='center', va='bottom',
                fontsize=8.5
            )

        ax2.set_xticks(x2)
        ax2.set_xticklabels(models, fontsize=10)
        ax2.set_ylim(0.75, 0.92)
        style_ax(ax2,
                title='NHANES real data\nBest calibration (ECE)\n'
                      'Causal-dominant regime',
                ylabel='Mean AUC-ROC')

    # ── Panel 3: UCI ───────────────────────────────────────────────────────
    ax3     = axes[2]
    uci_csv = os.path.join(ROOT, 'experiments', 'results',
                          'uci_heart', 'uci_summary.csv')
    if os.path.exists(uci_csv):
        uci    = pd.read_csv(uci_csv)
        models = ['ERM', 'IRM', 'CausTab']
        aucs   = [uci[uci['Model']==m]['Mean_AUC'].values[0]
                 for m in models]
        x3     = np.arange(len(models))

        bars = ax3.bar(x3, aucs, 0.5,
                      color=[COLORS[m] for m in models],
                      alpha=0.88, edgecolor='white')
        bars[2].set_edgecolor('black')
        bars[2].set_linewidth(2)

        for bar, v in zip(bars, aucs):
            ax3.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.006,
                f'{v:.3f}',
                ha='center', va='bottom',
                fontsize=9
            )

        ax3.set_xticks(x3)
        ax3.set_xticklabels(models, fontsize=10)
        ax3.set_ylim(0.4, 0.9)
        style_ax(ax3,
                title='UCI Heart Disease\nBoundary condition:\n'
                      'Extreme prevalence shift',
                ylabel='Mean AUC-ROC')

        ax3.text(
            0.5, 0.08,
            "CausTab underperforms when\n"
            "base rates differ dramatically",
            transform=ax3.transAxes,
            ha='center', fontsize=8.5,
            style='italic',
            bbox=dict(boxstyle='round,pad=0.4',
                     facecolor='#FFF9C4',
                     edgecolor='#F5A623',
                     alpha=0.95)
        )

    fig.tight_layout(pad=2.5)
    save(fig, 'fig9_summary.png')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("CausTab — Style B Plot Generator (NeurIPS/ICML)")
    print(f"Output: experiments/plots/publicationB/")
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
    print("ALL STYLE B FIGURES COMPLETE")
    print(f"Saved to: experiments/plots/publicationB/")
    print(f"{'='*60}")