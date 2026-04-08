"""
CausTab — Final Figures (SciencePlots)
Clean, minimal, consistent with top ML venues.
No bars. Thin lines. Light fonts. Generous spacing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scienceplots
import os
import sys

plt.style.use(['science', 'no-latex'])

plt.rcParams.update({
    'figure.dpi':           150,
    'savefig.dpi':          300,
    'savefig.bbox':         'tight',
    'savefig.pad_inches':   0.08,
    'legend.fontsize':      7,
    'axes.titlesize':       7.5,
    'axes.titlepad':        6,
    'axes.labelsize':       7,
    'xtick.labelsize':      6.5,
    'ytick.labelsize':      6.5,
    'lines.linewidth':      1.0,
    'lines.markersize':     3.5,
    'figure.titlesize':     8,
    'figure.titleweight':   'normal',
})

C = {
    'ERM':     '#0077BB',
    'IRM':     '#CC3311',
    'CausTab': '#009988',
}
ALPHA_FILL = 0.10
MARKERS = {'ERM': 'o', 'IRM': 's', 'CausTab': '^'}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, 'experiments', 'plots', 'science')
os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), facecolor='white')
    plt.close(fig)
    print(f"  {name}")

def make_fig(nrows, ncols, figsize, title,
             wspace=0.42, hspace=0.3):
    """
    Consistent figure factory.
    Every figure uses this — guarantees identical
    spacing, title placement, and breathing room.
    """
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        gridspec_kw={
            'wspace': wspace,
            'hspace': hspace,
            'top':    0.86,
            'bottom': 0.16,
            'left':   0.12,
            'right':  0.97,
        }
    )
    fig.suptitle(title, fontsize=8, y=1.02)
    return fig, axes

def annotate_point(ax, x, y, text, color,
                   dx=0.3, dy=0.02):
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        fontsize=6,
        color=color,
        arrowprops=dict(
            arrowstyle='->', lw=0.5,
            color=color, alpha=0.7
        ),
        va='center'
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — NHANES shift evidence
# ══════════════════════════════════════════════════════════════════════════════

def fig1():
    path = os.path.join(ROOT, 'data', 'tables',
                        'table3_correlation_shift.csv')
    if not os.path.exists(path):
        print("  SKIP fig1"); return

    df = pd.read_csv(path)
    nm = {
        'Age (years)':              'Age',
        'Gender (1=M, 2=F)':        'Gender',
        'Race/ethnicity':           'Race/ethnicity',
        'Income-to-poverty ratio':  'Income ratio',
        'Education level':          'Education',
        'Systolic BP 1 (mmHg)':     'Systolic BP 1',
        'Diastolic BP 1 (mmHg)':    'Diastolic BP 1',
        'Systolic BP 2 (mmHg)':     'Systolic BP 2',
        'Diastolic BP 2 (mmHg)':    'Diastolic BP 2',
        'BMI (kg/m\xb2)':           'BMI',
        'Waist circumference (cm)': 'Waist circ.',
    }
    df['Feature'] = df['Feature'].map(lambda x: nm.get(x, x))
    cycles = ['2011-12', '2013-14', '2015-16', '2017-18']
    avail  = [c for c in cycles if c in df.columns]
    df     = df.sort_values('Range', ascending=False)
    feats  = df['Feature'].tolist()
    mat    = df.set_index('Feature')[avail].values
    rng    = df['Range'].values

    fig, axes = make_fig(
        1, 2, (6.5, 3.4),
        'Distribution shift evidence in NHANES (2011--2018)',
        wspace=0.52
    )
    # Wider left margin for feature names
    fig.subplots_adjust(left=0.17, right=0.97,
                       top=0.86, bottom=0.14)

    # Heatmap
    ax = axes[0]
    im = ax.imshow(mat, cmap='RdBu_r',
                  vmin=-0.5, vmax=0.5, aspect='auto')
    ax.set_xticks(range(len(avail)))
    ax.set_xticklabels(avail, rotation=30,
                      ha='right', fontsize=6.5)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=6.5)
    ax.set_title('Correlation with outcome per NHANES cycle',
                fontsize=7.5)
    ax.tick_params(length=0)

    for i in range(len(feats)):
        for j in range(len(avail)):
            v = mat[i, j]
            ax.text(j, i, f'{v:.2f}',
                   ha='center', va='center',
                   fontsize=5.5,
                   color='white' if abs(v) > 0.28
                   else '#333333')

    cb = plt.colorbar(im, ax=ax,
                     fraction=0.03, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.set_label('r', fontsize=7)

    # Cleveland dot plot
    ax2   = axes[1]
    y_pos = np.arange(len(feats))
    colors_dot = ['#CC3311' if r > 0.02 else '#009988'
                  for r in rng]

    for i, (r, col) in enumerate(zip(rng, colors_dot)):
        ax2.plot([0, r], [i, i],
                color=col, lw=0.5, alpha=0.4)
        ax2.scatter(r, i, color=col,
                   s=16, zorder=3, lw=0)

    ax2.axvline(0.02, color='#888888', lw=0.6,
               ls='--', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(feats, fontsize=6.5)
    ax2.set_xlabel('Correlation range (max $-$ min)',
                  fontsize=7)
    ax2.set_title('Feature stability across cycles',
                 fontsize=7.5)
    ax2.set_xlim(-0.002, rng.max() + 0.012)
    ax2.tick_params(left=False)

    stable   = mpatches.Patch(color='#009988',
                              label='Stable')
    unstable = mpatches.Patch(color='#CC3311',
                              label='Unstable')
    thresh   = plt.Line2D([0],[0], ls='--', lw=0.6,
                         color='#888888',
                         label='$\\tau=0.02$')
    ax2.legend(handles=[stable, unstable, thresh],
              fontsize=6.5, loc='lower right',
              framealpha=0.9)

    save(fig, 'fig1_nhanes_shift.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — IRM failure
# ══════════════════════════════════════════════════════════════════════════════

def fig2():
    path = os.path.join(ROOT, 'experiments', 'results',
                        'irm_analysis',
                        'spurious_sweep_results.csv')
    if not os.path.exists(path):
        print("  SKIP fig2"); return

    df = pd.read_csv(path)
    ss = sorted(df['Spurious_strength'].unique())

    fig, axes = make_fig(
        1, 2, (6.5, 2.8),
        'IRM failure on tabular data: '
        'performance vs spurious feature strength'
    )

    for ax, metric, ylabel, title in zip(
        axes,
        ['Mean_AUC', 'Std_AUC'],
        ['Mean AUC-ROC', 'Std of AUC (5 seeds)'],
        ['Predictive performance',
         'Stability (lower = more reliable)']
    ):
        for m in ['ERM', 'IRM', 'CausTab']:
            sub  = df[df['Model'] == m]
            vals = [sub[sub['Spurious_strength']==s
                       ][metric].values[0] for s in ss]
            ax.plot(ss, vals,
                   marker=MARKERS[m], lw=1.0, ms=3.5,
                   color=C[m], label=m)

            if metric == 'Mean_AUC':
                stds = [sub[sub['Spurious_strength']==s
                           ]['Std_AUC'].values[0]
                       for s in ss]
                ax.fill_between(
                    ss,
                    [v-s for v,s in zip(vals,stds)],
                    [v+s for v,s in zip(vals,stds)],
                    alpha=ALPHA_FILL, color=C[m]
                )

        if metric == 'Mean_AUC':
            ax.axhline(0.5, color='#aaaaaa',
                      lw=0.5, ls=':', label='Random')
            ax.set_ylim(0.44, 1.02)

            irm  = df[df['Model']=='IRM']
            erm  = df[df['Model']=='ERM']
            gaps = [
                irm[irm['Spurious_strength']==s
                   ]['Mean_AUC'].values[0] -
                erm[erm['Spurious_strength']==s
                   ]['Mean_AUC'].values[0]
                for s in ss
            ]
            wi = np.argmin(gaps)
            ws = ss[wi]
            wv = irm[irm['Spurious_strength']==ws
                     ]['Mean_AUC'].values[0]
            annotate_point(ax, ws, wv,
                          f'max gap\n{gaps[wi]:+.3f}',
                          C['IRM'], dx=-0.8, dy=0.08)

        ax.set_xlabel('Spurious feature strength', fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_title(title, fontsize=7.5)
        ax.legend(fontsize=7, framealpha=0.9)

    save(fig, 'fig2_irm_failure.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Synthetic regimes
# ══════════════════════════════════════════════════════════════════════════════

def fig3():
    path = os.path.join(ROOT, 'experiments', 'results',
                        'synthetic', 'synthetic_results.csv')
    if not os.path.exists(path):
        print("  SKIP fig3"); return

    df      = pd.read_csv(path)
    regimes = list(df['Regime'].unique())
    xlabs   = [
        'Causal dominant\n(SDI=1.67)',
        'Mixed\n(SDI=9.62)',
        'Spurious dominant\n(SDI=48.08)',
    ]
    x = np.arange(len(regimes))

    fig, ax = make_fig(
        1, 1, (4.5, 3.0),
        'Synthetic experiment: three spurious-correlation regimes'
    )
    # Single axis — unwrap
    if hasattr(ax, '__len__'):
        ax = ax.flat[0]

    for m in ['ERM', 'IRM', 'CausTab']:
        sub   = df[df['Model'] == m]
        means = [sub[sub['Regime']==r
                    ]['Mean_AUC'].values[0] for r in regimes]
        stds  = [sub[sub['Regime']==r
                    ]['Std_AUC'].values[0] for r in regimes]

        ax.plot(x, means,
               marker=MARKERS[m], lw=1.0, ms=4,
               color=C[m], label=m, zorder=3)
        ax.fill_between(
            x,
            [v-s for v,s in zip(means,stds)],
            [v+s for v,s in zip(means,stds)],
            alpha=ALPHA_FILL, color=C[m]
        )

        offsets = {'ERM': 0.010,
                  'IRM': -0.024,
                  'CausTab': 0.010}
        off = offsets[m]
        for xi, v in enumerate(means):
            ax.text(xi, v + off,
                   f'{v:.3f}',
                   ha='center', va='bottom',
                   fontsize=6, color=C[m])

    ax.set_xticks(x)
    ax.set_xticklabels(xlabs, fontsize=7)
    ax.set_ylabel('Mean AUC-ROC', fontsize=7)
    ax.set_ylim(0.55, 1.07)
    ax.axhline(0.5, color='#cccccc', lw=0.4, ls=':')
    ax.legend(fontsize=7, loc='lower left', framealpha=0.9)

    save(fig, 'fig3_synthetic_regimes.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — NHANES temporal evaluation
# ══════════════════════════════════════════════════════════════════════════════

def fig4():
    path = os.path.join(ROOT, 'experiments', 'results',
                        'temporal_split',
                        'all_temporal_results.csv')
    if not os.path.exists(path):
        print("  SKIP fig4"); return

    df     = pd.read_csv(path)
    splits = sorted(df['Split'].unique())
    ls_map = {splits[0]: '-', splits[1]: '--'}
    mk_map = {splits[0]: 'o', splits[1]: 's'}

    fig, axes = make_fig(
        1, 2, (6.5, 2.8),
        'NHANES temporal forward-chaining evaluation'
    )

    for ax, metric, ylabel in zip(
        axes, ['AUC', 'ECE'],
        ['AUC-ROC', 'ECE ($\\downarrow$)']
    ):
        for split in splits:
            sub   = df[df['Split'] == split]
            envs  = sorted(sub['Environment'].unique())
            short = split.replace('Split_', 'S')

            for m in ['ERM', 'IRM', 'CausTab']:
                ms    = sub[sub['Model'] == m]
                elabs = [e for e in envs
                        if e in ms['Environment'].values]
                vals  = [ms[ms['Environment']==e
                            ][metric].values[0]
                        for e in elabs]

                ax.plot(elabs, vals,
                       marker=mk_map[split],
                       lw=1.0, ms=3.5,
                       ls=ls_map[split],
                       color=C[m], alpha=0.9,
                       label=f'{m} ({short})')

                if (metric == 'AUC' and
                        'AUC_CI_lo' in ms.columns):
                    lo = [ms[ms['Environment']==e
                             ]['AUC_CI_lo'].values[0]
                         for e in elabs]
                    hi = [ms[ms['Environment']==e
                             ]['AUC_CI_hi'].values[0]
                         for e in elabs]
                    ax.fill_between(elabs, lo, hi,
                                   alpha=ALPHA_FILL,
                                   color=C[m])

        ax.set_xlabel('Test environment', fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_title(ylabel, fontsize=7.5)
        ax.legend(fontsize=6, loc='best',
                 framealpha=0.9, ncol=1)
        ax.tick_params(axis='x', rotation=15)

    save(fig, 'fig4_nhanes_temporal.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Calibration ECE
# ══════════════════════════════════════════════════════════════════════════════

def fig5():
    files = {
        'Standard':   os.path.join(
            ROOT, 'experiments', 'results',
            'evaluation_results.csv'),
        'Temporal B': os.path.join(
            ROOT, 'experiments', 'results',
            'temporal_split', 'split_b_results.csv'),
        'Temporal C': os.path.join(
            ROOT, 'experiments', 'results',
            'temporal_split', 'split_c_results.csv'),
    }
    if not os.path.exists(list(files.values())[0]):
        print("  SKIP fig5"); return

    fig, axes = make_fig(
        1, 2, (6.5, 2.8),
        'Probability calibration (ECE) '
        'across NHANES experiments'
    )

    # Left: ECE per cycle
    ax  = axes[0]
    std = pd.read_csv(files['Standard'])
    envs = ['2011-12', '2013-14', '2015-16', '2017-18']
    x    = np.arange(len(envs))

    for m in ['ERM', 'IRM', 'CausTab']:
        sub  = std[std['Model'] == m]
        vals = [sub[sub['Environment']==e
                   ]['ECE'].values[0]
               if e in sub['Environment'].values
               else np.nan for e in envs]
        ax.plot(x, vals,
               marker=MARKERS[m], lw=1.0, ms=3.5,
               color=C[m], label=m)

    ax.set_xticks(x)
    ax.set_xticklabels(envs, fontsize=6.5,
                      rotation=20, ha='right')
    ax.set_ylabel('ECE', fontsize=7)
    ax.set_xlabel('Survey cycle', fontsize=7)
    ax.set_title('ECE per cycle (standard split)', fontsize=7.5)
    ax.legend(fontsize=7, framealpha=0.9)

    # Right: mean ECE across experiments
    ax2       = axes[1]
    exp_labs  = list(files.keys())
    x2        = np.arange(len(exp_labs))
    ece_data  = {m: [] for m in C}

    for fpath in files.values():
        if not os.path.exists(fpath):
            for m in ece_data:
                ece_data[m].append(np.nan)
            continue
        sub = pd.read_csv(fpath)
        for m in ece_data:
            ms = sub[sub['Model']==m] \
                if 'Model' in sub.columns \
                else pd.DataFrame()
            ece_data[m].append(
                ms['ECE'].mean()
                if len(ms) > 0 and 'ECE' in ms.columns
                else np.nan
            )

    for m in ['ERM', 'IRM', 'CausTab']:
        vals = ece_data[m]
        ax2.plot(x2, vals,
                marker=MARKERS[m], lw=1.0, ms=3.5,
                color=C[m], label=m)
        for xi, v in enumerate(vals):
            if not np.isnan(v):
                ax2.text(xi, v + 0.0005,
                        f'{v:.4f}',
                        ha='center', va='bottom',
                        fontsize=6, color=C[m])

    ax2.set_xticks(x2)
    ax2.set_xticklabels(exp_labs, fontsize=7)
    ax2.set_ylabel('Mean ECE', fontsize=7)
    ax2.set_title('Mean ECE across all experiments',
                 fontsize=7.5)
    ax2.legend(fontsize=7, framealpha=0.9)

    save(fig, 'fig5_calibration.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — SDI validation
# ══════════════════════════════════════════════════════════════════════════════

def fig6():
    path = os.path.join(ROOT, 'experiments', 'results',
                        'synthetic', 'synthetic_results.csv')
    if not os.path.exists(path):
        print("  SKIP fig6"); return

    df = pd.read_csv(path)

    fig, ax = make_fig(
        1, 1, (4.0, 3.0),
        'SDI validation: higher SDI predicts '
        'larger CausTab advantage'
    )
    if hasattr(ax, '__len__'):
        ax = ax.flat[0]

    rcolors = ['#009988', '#F28E2B', '#CC3311']
    rlabs   = {
        'Regime_1_Causal_Dominant':   'R1: Causal',
        'Regime_2_Mixed':             'R2: Mixed',
        'Regime_3_Spurious_Dominant': 'R3: Spurious',
    }
    offsets = {
        'Regime_1_Causal_Dominant':   (3,   0.003),
        'Regime_2_Mixed':             (3,   0.003),
        'Regime_3_Spurious_Dominant': (-12, 0.003),
    }

    for regime, rc in zip(df['Regime'].unique(), rcolors):
        sub = df[df['Regime'] == regime]
        sdi = sub['SDI'].iloc[0]
        ct  = sub[sub['Model']=='CausTab']['Mean_AUC'].values[0]
        er  = sub[sub['Model']=='ERM']['Mean_AUC'].values[0]
        std = sub[sub['Model']=='CausTab']['Std_AUC'].values[0]
        adv = ct - er
        lab = rlabs.get(regime, regime)
        ox, oy = offsets.get(regime, (3, 0.003))

        ax.scatter(sdi, adv, color=rc, s=28,
                  zorder=4, lw=0)
        ax.errorbar(sdi, adv, yerr=std,
                   fmt='none', ecolor=rc,
                   elinewidth=0.6, capsize=2,
                   capthick=0.6, alpha=0.5)
        ax.annotate(lab,
                   xy=(sdi, adv),
                   xytext=(sdi+ox, adv+oy),
                   fontsize=6.5, color=rc,
                   arrowprops=dict(
                       arrowstyle='-', lw=0.4,
                       color=rc, alpha=0.6))

    ax.scatter(1.7, 0.0003, color='#9467BD',
              s=22, marker='D', zorder=4, lw=0,
              label='NHANES')
    ax.annotate('NHANES',
               xy=(1.7, 0.0003),
               xytext=(4.5, 0.004),
               fontsize=6.5, color='#9467BD',
               arrowprops=dict(
                   arrowstyle='-', lw=0.4,
                   color='#9467BD'))

    ax.axhline(0, color='#aaaaaa', lw=0.5,
              ls='--', alpha=0.7)
    ax.axhspan(-0.001, 0.001, alpha=0.05, color='gray')
    ax.set_xlabel('Spurious Dominance Index (SDI)', fontsize=7)
    ax.set_ylabel('CausTab $-$ ERM (AUC)', fontsize=7)
    ax.legend(fontsize=6.5, loc='upper left', framealpha=0.9)

    save(fig, 'fig6_sdi_validation.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Ablation study
# ══════════════════════════════════════════════════════════════════════════════

def fig7():
    syn_p = os.path.join(ROOT, 'experiments', 'results',
                         'ablation',
                         'ablation_synthetic_results.csv')
    nha_p = os.path.join(ROOT, 'experiments', 'results',
                         'ablation',
                         'ablation_nhanes_results.csv')
    if not os.path.exists(syn_p):
        print("  SKIP fig7"); return

    syn = pd.read_csv(syn_p)
    nha = pd.read_csv(nha_p) if os.path.exists(nha_p) else None

    variants = [
        'CausTab_Full', 'CausTab_NoAnneal',
        'CausTab_NoWarmup', 'CausTab_MeanPenalty',
        'CausTab_NoPenalty',
    ]
    vlabs = {
        'CausTab_Full':        'Full (proposed)',
        'CausTab_NoAnneal':    '$-$Annealing',
        'CausTab_NoWarmup':    '$-$Warmup',
        'CausTab_MeanPenalty': 'Mean penalty',
        'CausTab_NoPenalty':   '$-$Penalty (ERM)',
    }
    vcols = {
        'CausTab_Full':        C['CausTab'],
        'CausTab_NoAnneal':    '#4878CF',
        'CausTab_NoWarmup':    '#9467BD',
        'CausTab_MeanPenalty': '#E6A817',
        'CausTab_NoPenalty':   C['IRM'],
    }
    vmk = {
        'CausTab_Full':        '^',
        'CausTab_NoAnneal':    'o',
        'CausTab_NoWarmup':    's',
        'CausTab_MeanPenalty': 'D',
        'CausTab_NoPenalty':   'v',
    }

    regimes = list(syn['Regime'].unique())
    rlabs   = ['R1: Causal', 'R2: Mixed', 'R3: Spurious']
    x       = np.arange(len(regimes))
    ncols   = 2 if nha is not None else 1

    fig, axes = make_fig(
        1, ncols, (3.8*ncols, 3.0),
        'Ablation study: each variant removes one component',
        wspace=0.40
    )
    if ncols == 1:
        axes = [axes]

    # Synthetic panel
    ax = axes[0]
    for v in variants:
        sub   = syn[syn['Variant'] == v]
        means = [sub[sub['Regime']==r]['Mean_AUC'].values[0]
                if len(sub[sub['Regime']==r]) > 0 else np.nan
                for r in regimes]
        stds  = [sub[sub['Regime']==r]['Std_AUC'].values[0]
                if len(sub[sub['Regime']==r]) > 0 else 0
                for r in regimes]
        lw   = 1.2 if v == 'CausTab_Full' else 0.7
        alph = 1.0 if v == 'CausTab_Full' else 0.7

        ax.plot(x, means,
               marker=vmk[v], lw=lw, ms=3.5,
               color=vcols[v], label=vlabs[v],
               alpha=alph,
               zorder=3 if v=='CausTab_Full' else 2)
        ax.fill_between(
            x,
            [m-s for m,s in zip(means,stds)],
            [m+s for m,s in zip(means,stds)],
            alpha=0.06, color=vcols[v]
        )

    ax.set_xticks(x)
    ax.set_xticklabels(rlabs, fontsize=7)
    ax.set_ylabel('Mean AUC-ROC', fontsize=7)
    ax.set_title('Synthetic data', fontsize=7.5)
    ax.legend(fontsize=6, framealpha=0.9, loc='lower left')

    # NHANES panel
    if nha is not None:
        ax2 = axes[1]
        x2  = np.arange(2)

        for v in variants:
            sub = nha[nha['Variant'] == v]
            if len(sub) == 0:
                continue
            auc  = sub['Mean_AUC'].values[0]
            ece  = sub['Mean_ECE'].values[0]
            lw   = 1.2 if v == 'CausTab_Full' else 0.7
            alph = 1.0 if v == 'CausTab_Full' else 0.7

            ax2.plot(x2, [auc, ece],
                    marker=vmk[v], lw=lw, ms=3.5,
                    color=vcols[v], label=vlabs[v],
                    alpha=alph,
                    zorder=3 if v=='CausTab_Full' else 2)
            for xi, val in enumerate([auc, ece]):
                ax2.text(xi, val + 0.0003,
                        f'{val:.4f}',
                        ha='center', va='bottom',
                        fontsize=5.5, color=vcols[v],
                        alpha=alph)

        ax2.set_xticks(x2)
        ax2.set_xticklabels(['AUC-ROC', 'ECE'], fontsize=7)
        ax2.set_title('NHANES temporal split B', fontsize=7.5)
        ax2.legend(fontsize=6, framealpha=0.9,
                  loc='center right')

    save(fig, 'fig7_ablation.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 — UCI Heart Disease
# ══════════════════════════════════════════════════════════════════════════════

def fig8():
    path = os.path.join(ROOT, 'experiments', 'results',
                        'uci_heart', 'uci_loocv_results.csv')
    if not os.path.exists(path):
        print("  SKIP fig8"); return

    df        = pd.read_csv(path)
    hospitals = list(df['Test_Hospital'].unique())
    short_h   = [h.split('(')[0].strip() for h in hospitals]
    x         = np.arange(len(hospitals))

    fig, axes = make_fig(
        1, 3, (7.5, 2.8),
        'UCI Heart Disease: leave-one-hospital-out '
        'evaluation (institutional shift)',
        wspace=0.46
    )

    for ax, metric, ylabel in zip(
        axes, ['AUC', 'F1', 'ECE'],
        ['AUC-ROC', 'F1 score', 'ECE ($\\downarrow$)']
    ):
        for m in ['ERM', 'IRM', 'CausTab']:
            sub  = df[df['Model'] == m]
            vals = [sub[sub['Test_Hospital']==h
                       ][metric].values[0]
                   for h in hospitals]
            ax.plot(x, vals,
                   marker=MARKERS[m], lw=1.0, ms=3.5,
                   color=C[m], label=m)

            if (metric == 'AUC' and
                    'AUC_CI_lo' in sub.columns):
                lo = [sub[sub['Test_Hospital']==h
                         ]['AUC_CI_lo'].values[0]
                     for h in hospitals]
                hi = [sub[sub['Test_Hospital']==h
                         ]['AUC_CI_hi'].values[0]
                     for h in hospitals]
                ax.fill_between(x, lo, hi,
                               alpha=ALPHA_FILL,
                               color=C[m])

        ax.set_xticks(x)
        ax.set_xticklabels(short_h, fontsize=6.5,
                          rotation=18, ha='right')
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_title(ylabel, fontsize=7.5)
        ax.legend(fontsize=6.5, framealpha=0.9)
        ax.set_ylim(0.15, 1.05)

    fig.text(
        0.5, 0.01,
        'Note: CausTab underperforms ERM on Cleveland '
        'due to extreme prevalence heterogeneity '
        '(boundary condition, see Sec. 6).',
        ha='center', fontsize=6.5,
        style='italic', color='#555555'
    )

    save(fig, 'fig8_uci_results.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 9 — Summary
# ══════════════════════════════════════════════════════════════════════════════

def fig9():
    fig, axes = make_fig(
        1, 3, (7.5, 2.8),
        'When does invariant learning help? '
        'Three experiments, one answer.',
        wspace=0.46
    )

    # Panel 1: Synthetic
    ax      = axes[0]
    syn_p   = os.path.join(ROOT, 'experiments', 'results',
                           'synthetic', 'synthetic_results.csv')
    if os.path.exists(syn_p):
        syn     = pd.read_csv(syn_p)
        regimes = list(syn['Regime'].unique())
        x       = np.arange(len(regimes))
        xlabs   = ['R1\nCausal', 'R2\nMixed', 'R3\nSpurious']

        for m in ['ERM', 'IRM', 'CausTab']:
            sub   = syn[syn['Model'] == m]
            means = [sub[sub['Regime']==r
                        ]['Mean_AUC'].values[0]
                    for r in regimes]
            stds  = [sub[sub['Regime']==r
                        ]['Std_AUC'].values[0]
                    for r in regimes]
            ax.plot(x, means,
                   marker=MARKERS[m], lw=1.0, ms=3.5,
                   color=C[m], label=m)
            ax.fill_between(
                x,
                [v-s for v,s in zip(means,stds)],
                [v+s for v,s in zip(means,stds)],
                alpha=ALPHA_FILL, color=C[m]
            )

        ax.set_xticks(x)
        ax.set_xticklabels(xlabs, fontsize=7)
        ax.set_ylim(0.55, 1.04)
        ax.set_ylabel('Mean AUC-ROC', fontsize=7)
        ax.set_title('Synthetic: CausTab $\\geq$ ERM always',
                    fontsize=7.5)
        ax.legend(fontsize=6.5, loc='lower left',
                 framealpha=0.9)

    # Panel 2: NHANES ECE
    ax2   = axes[1]
    std_p = os.path.join(ROOT, 'experiments', 'results',
                         'evaluation_results.csv')
    if os.path.exists(std_p):
        std  = pd.read_csv(std_p)
        envs = ['2011-12', '2013-14', '2015-16', '2017-18']
        x2   = np.arange(len(envs))

        for m in ['ERM', 'IRM', 'CausTab']:
            sub  = std[std['Model'] == m]
            eces = [sub[sub['Environment']==e
                       ]['ECE'].values[0]
                   if e in sub['Environment'].values
                   else np.nan for e in envs]
            ax2.plot(x2, eces,
                    marker=MARKERS[m], lw=1.0, ms=3.5,
                    color=C[m], label=m)

        ax2.set_xticks(x2)
        ax2.set_xticklabels(envs, fontsize=6.5,
                           rotation=18, ha='right')
        ax2.set_ylabel('ECE', fontsize=7)
        ax2.set_title('NHANES: best calibration (ECE)',
                     fontsize=7.5)
        ax2.legend(fontsize=6.5, framealpha=0.9)

    # Panel 3: UCI AUC
    ax3   = axes[2]
    uci_p = os.path.join(ROOT, 'experiments', 'results',
                         'uci_heart', 'uci_loocv_results.csv')
    if os.path.exists(uci_p):
        uci       = pd.read_csv(uci_p)
        hospitals = list(uci['Test_Hospital'].unique())
        short_h   = [h.split('(')[0].strip()
                    for h in hospitals]
        x3        = np.arange(len(hospitals))

        for m in ['ERM', 'IRM', 'CausTab']:
            sub  = uci[uci['Model'] == m]
            aucs = [sub[sub['Test_Hospital']==h
                       ]['AUC'].values[0]
                   for h in hospitals]
            ax3.plot(x3, aucs,
                    marker=MARKERS[m], lw=1.0, ms=3.5,
                    color=C[m], label=m)

        ax3.set_xticks(x3)
        ax3.set_xticklabels(short_h, fontsize=6.5,
                           rotation=18, ha='right')
        ax3.set_ylim(0.2, 1.0)
        ax3.set_ylabel('AUC-ROC', fontsize=7)
        ax3.set_title('UCI: boundary condition identified',
                     fontsize=7.5)
        ax3.legend(fontsize=6.5, framealpha=0.9)
        ax3.text(
            0.5, 0.08,
            'CausTab $<$ ERM:\nextreme prevalence shift',
            transform=ax3.transAxes,
            ha='center', fontsize=6,
            style='italic', color='#666666',
            bbox=dict(boxstyle='round,pad=0.25',
                     facecolor='#fffbe6',
                     edgecolor='#ddaa00',
                     alpha=0.9, lw=0.4)
        )

    save(fig, 'fig9_summary.png')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=' * 50)
    print('CausTab — final figures (SciencePlots)')
    print('Output: experiments/plots/science/')
    print('=' * 50)
    print()
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    fig9()
    print()
    print('Done. Open experiments/plots/science/')