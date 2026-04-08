"""
CausTab — Data Exploration
Understand the dataset before we model anything.
Rule: never trust data you haven't looked at.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Load the master dataset ───────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nhanes_master.csv")
df = pd.read_csv(DATA_PATH)

# Plain English: load our clean CSV into a dataframe called df
# A dataframe is just a table — rows are people, columns are variables

print("=" * 60)
print("CAUSTAB — DATA EXPLORATION")
print("=" * 60)

# ── 1. Basic shape ────────────────────────────────────────────────────────────
print(f"\n[1] SHAPE")
print(f"    Rows (participants): {len(df):,}")
print(f"    Columns (variables): {df.shape[1]}")
print(f"    Environments: {sorted(df['environment'].unique())}")

# ── 2. Variable types and missing values ──────────────────────────────────────
print(f"\n[2] VARIABLES")
print(f"    {'Column':<15} {'Type':<10} {'Missing':>8} {'Min':>8} {'Max':>8}")
print(f"    {'-'*55}")
for col in df.columns:
    if col in ['SEQN', 'environment', 'hypertension']:
        continue
    missing = df[col].isna().sum()
    print(f"    {col:<15} {str(df[col].dtype):<10} "
          f"{missing:>8} {df[col].min():>8.1f} {df[col].max():>8.1f}")

# ── 3. Outcome distribution ───────────────────────────────────────────────────
print(f"\n[3] OUTCOME — Hypertension")
print(f"    0 = No hypertension: {(df['hypertension']==0).sum():,} "
      f"({(df['hypertension']==0).mean()*100:.1f}%)")
print(f"    1 = Has hypertension: {(df['hypertension']==1).sum():,} "
      f"({(df['hypertension']==1).mean()*100:.1f}%)")

# ── 4. THIS IS THE CORE OF OUR PAPER ─────────────────────────────────────────
# We need to show that distributions SHIFT across environments
# If they don't shift, our paper has no motivation
# If they do shift, we have evidence the problem is real
print(f"\n[4] DISTRIBUTION SHIFT EVIDENCE — The heart of our paper")
print(f"    Hypertension prevalence per environment:")
prev = df.groupby('environment')['hypertension'].mean() * 100
for env, p in prev.items():
    bar = '█' * int(p / 2)
    print(f"    {env}: {p:.1f}%  {bar}")

print(f"\n    Key feature means per environment:")
features = ['RIDAGEYR', 'BMXBMI', 'BPXSY1', 'INDFMPIR']
feature_names = {
    'RIDAGEYR': 'Age',
    'BMXBMI':   'BMI',
    'BPXSY1':   'Systolic BP',
    'INDFMPIR': 'Income ratio'
}
env_means = df.groupby('environment')[features].mean()
print(f"\n    {'Feature':<15}", end="")
for env in sorted(df['environment'].unique()):
    print(f" {env:>12}", end="")
print()
print(f"    {'-'*60}")
for feat in features:
    print(f"    {feature_names[feat]:<15}", end="")
    for env in sorted(df['environment'].unique()):
        print(f" {env_means.loc[env, feat]:>12.2f}", end="")
    print()

# ── 5. Correlation structure per environment ──────────────────────────────────
# This is KEY — if correlations between features and outcome change
# across environments, that IS the spurious correlation problem we're solving
print(f"\n[5] CORRELATION SHIFT — Spurious vs causal signal")
print(f"    Correlation of each feature with hypertension, per environment:")
print(f"    (If these numbers change a lot across cycles = distribution shift is real)")
print()
feat_cols = ['RIDAGEYR', 'RIAGENDR', 'BMXBMI', 'BMXWAIST',
             'BPXSY1', 'BPXDI1', 'INDFMPIR', 'DMDEDUC2']
print(f"    {'Feature':<15}", end="")
for env in sorted(df['environment'].unique()):
    print(f" {env:>12}", end="")
print(f"  {'Range':>8}")
print(f"    {'-'*72}")
for feat in feat_cols:
    corrs = []
    for env in sorted(df['environment'].unique()):
        sub = df[df['environment'] == env]
        c = sub[feat].corr(sub['hypertension'])
        corrs.append(c)
    rng = max(corrs) - min(corrs)
    print(f"    {feat:<15}", end="")
    for c in corrs:
        print(f" {c:>12.4f}", end="")
    marker = " ← SHIFTS" if rng > 0.02 else ""
    print(f"  {rng:>8.4f}{marker}")

# ── 6. Age distribution — known causal factor ─────────────────────────────────
print(f"\n[6] AGE DISTRIBUTION per environment")
print(f"    Age is a known causal driver of hypertension.")
print(f"    If age distribution shifts, that explains some outcome shift.")
age_stats = df.groupby('environment')['RIDAGEYR'].agg(['mean','std','min','max'])
print(f"\n    {'Cycle':<12} {'Mean age':>10} {'Std':>8} {'Min':>6} {'Max':>6}")
print(f"    {'-'*45}")
for env, row in age_stats.iterrows():
    print(f"    {env:<12} {row['mean']:>10.1f} {row['std']:>8.1f} "
          f"{row['min']:>6.0f} {row['max']:>6.0f}")

# ── 7. Save plots ──────────────────────────────────────────────────────────────
print(f"\n[7] SAVING PLOTS to data/plots/")
plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(plots_dir, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("CausTab — Distribution Shift Evidence in NHANES", fontsize=14)

# Plot 1: Hypertension prevalence across cycles
envs = sorted(df['environment'].unique())
prevs = [df[df['environment']==e]['hypertension'].mean()*100 for e in envs]
axes[0,0].bar(envs, prevs, color=['#4C72B0','#DD8452','#55A868','#C44E52'])
axes[0,0].set_title("Hypertension prevalence per cycle")
axes[0,0].set_ylabel("Prevalence (%)")
axes[0,0].set_ylim(0, 50)
for i, v in enumerate(prevs):
    axes[0,0].text(i, v+0.5, f"{v:.1f}%", ha='center', fontsize=10)

# Plot 2: Age distribution across cycles
for env in envs:
    sub = df[df['environment']==env]['RIDAGEYR']
    axes[0,1].hist(sub, bins=20, alpha=0.5, label=env, density=True)
axes[0,1].set_title("Age distribution per cycle")
axes[0,1].set_xlabel("Age (years)")
axes[0,1].set_ylabel("Density")
axes[0,1].legend(fontsize=8)

# Plot 3: Systolic BP distribution across cycles
for env in envs:
    sub = df[df['environment']==env]['BPXSY1']
    axes[1,0].hist(sub, bins=30, alpha=0.5, label=env, density=True)
axes[1,0].set_title("Systolic BP distribution per cycle")
axes[1,0].set_xlabel("Systolic BP (mmHg)")
axes[1,0].set_ylabel("Density")
axes[1,0].legend(fontsize=8)

# Plot 4: Correlation shift heatmap
corr_data = {}
for env in envs:
    sub = df[df['environment']==env]
    corr_data[env] = [sub[f].corr(sub['hypertension']) for f in feat_cols]
corr_df = pd.DataFrame(corr_data, index=feat_cols)
sns.heatmap(corr_df, annot=True, fmt=".3f", cmap="RdBu_r",
            center=0, ax=axes[1,1], cbar_kws={'shrink':0.8})
axes[1,1].set_title("Feature-outcome correlations per cycle\n(variation = distribution shift)")

plt.tight_layout()
plot_path = os.path.join(plots_dir, "distribution_shift_evidence.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"    Saved: {plot_path}")
plt.close()

# ── 8. Save tables ─────────────────────────────────────────────────────────────
print(f"\n[8] SAVING TABLES to data/tables/")
tables_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
os.makedirs(tables_dir, exist_ok=True)

# ── Table 1: Dataset summary ───────────────────────────────────────────────────
t1 = df.groupby('environment').agg(
    N=('SEQN', 'count'),
    Hypertension_N=('hypertension', 'sum'),
    Hypertension_pct=('hypertension', lambda x: round(x.mean()*100, 1))
).reset_index()
t1.columns = ['Cycle', 'N', 'Hypertension (N)', 'Hypertension (%)']
t1.to_csv(os.path.join(tables_dir, "table1_dataset_summary.csv"), index=False)
with open(os.path.join(tables_dir, "table1_dataset_summary.txt"), 'w') as f:
    f.write("Table 1. Dataset summary by NHANES survey cycle\n")
    f.write("="*55 + "\n")
    f.write(f"{'Cycle':<12} {'N':>8} {'Hypert. N':>12} {'Hypert. %':>12}\n")
    f.write("-"*55 + "\n")
    for _, row in t1.iterrows():
        f.write(f"{row['Cycle']:<12} {int(row['N']):>8,} "
                f"{int(row['Hypertension (N)']):>12,} "
                f"{row['Hypertension (%)']:>11.1f}%\n")
    f.write("-"*55 + "\n")
    f.write(f"{'Total':<12} {len(df):>8,} "
            f"{int(df['hypertension'].sum()):>12,} "
            f"{df['hypertension'].mean()*100:>11.1f}%\n")
print(f"    Saved: table1_dataset_summary")

# ── Table 2: Feature means per environment ─────────────────────────────────────
feat_labels = {
    'RIDAGEYR':  'Age (years)',
    'RIAGENDR':  'Gender (1=M, 2=F)',
    'BMXBMI':    'BMI (kg/m²)',
    'BMXWAIST':  'Waist circumference (cm)',
    'BPXSY1':    'Systolic BP 1 (mmHg)',
    'BPXDI1':    'Diastolic BP 1 (mmHg)',
    'BPXSY2':    'Systolic BP 2 (mmHg)',
    'BPXDI2':    'Diastolic BP 2 (mmHg)',
    'INDFMPIR':  'Income-to-poverty ratio',
    'DMDEDUC2':  'Education level',
    'RIDRETH3':  'Race/ethnicity'
}
envs_sorted = sorted(df['environment'].unique())
feat_cols_all = list(feat_labels.keys())
t2_rows = []
for feat in feat_cols_all:
    row = {'Feature': feat_labels[feat]}
    for env in envs_sorted:
        sub = df[df['environment']==env][feat]
        row[env] = f"{sub.mean():.2f} ± {sub.std():.2f}"
    t2_rows.append(row)
t2 = pd.DataFrame(t2_rows)
t2.to_csv(os.path.join(tables_dir, "table2_feature_means.csv"), index=False)
with open(os.path.join(tables_dir, "table2_feature_means.txt"), 'w') as f:
    f.write("Table 2. Feature means (± SD) by NHANES survey cycle\n")
    f.write("="*75 + "\n")
    f.write(f"{'Feature':<28}")
    for env in envs_sorted:
        f.write(f" {env:>16}")
    f.write("\n" + "-"*75 + "\n")
    for _, row in t2.iterrows():
        f.write(f"{row['Feature']:<28}")
        for env in envs_sorted:
            f.write(f" {row[env]:>16}")
        f.write("\n")
print(f"    Saved: table2_feature_means")

# ── Table 3: Correlation shift ─────────────────────────────────────────────────
# This is the smoking gun table — goes in the paper as a key result
t3_rows = []
for feat in feat_cols_all:
    row = {'Feature': feat_labels[feat]}
    corrs = []
    for env in envs_sorted:
        sub = df[df['environment']==env]
        c = sub[feat].corr(sub['hypertension'])
        row[env] = round(c, 4)
        corrs.append(c)
    row['Range'] = round(max(corrs) - min(corrs), 4)
    row['Stable'] = 'Causal' if row['Range'] < 0.02 else 'Spurious/mixed'
    t3_rows.append(row)
t3 = pd.DataFrame(t3_rows)
t3.to_csv(os.path.join(tables_dir, "table3_correlation_shift.csv"), index=False)
with open(os.path.join(tables_dir, "table3_correlation_shift.txt"), 'w') as f:
    f.write("Table 3. Feature-outcome correlation by NHANES survey cycle\n")
    f.write("Note: Range = max - min correlation across cycles.\n")
    f.write("      Low range = causally stable. High range = spurious/mixed.\n")
    f.write("="*85 + "\n")
    f.write(f"{'Feature':<28}")
    for env in envs_sorted:
        f.write(f" {env:>10}")
    f.write(f" {'Range':>8}  {'Signal type'}\n")
    f.write("-"*85 + "\n")
    for _, row in t3.iterrows():
        f.write(f"{row['Feature']:<28}")
        for env in envs_sorted:
            f.write(f" {row[env]:>10.4f}")
        marker = " **" if row['Range'] > 0.02 else "   "
        f.write(f" {row['Range']:>8.4f}{marker}  {row['Stable']}\n")
    f.write("-"*85 + "\n")
    f.write("** Flags features with correlation range > 0.02 across cycles\n")
print(f"    Saved: table3_correlation_shift  ← KEY TABLE for paper")

# ── Table 4: Age distribution per environment ──────────────────────────────────
t4 = df.groupby('environment')['RIDAGEYR'].agg(
    Mean='mean', SD='std', Median='median', Min='min', Max='max'
).round(1).reset_index()
t4.columns = ['Cycle', 'Mean', 'SD', 'Median', 'Min', 'Max']
t4.to_csv(os.path.join(tables_dir, "table4_age_distribution.csv"), index=False)
with open(os.path.join(tables_dir, "table4_age_distribution.txt"), 'w') as f:
    f.write("Table 4. Age distribution by NHANES survey cycle\n")
    f.write("="*55 + "\n")
    f.write(f"{'Cycle':<12} {'Mean':>8} {'SD':>8} "
            f"{'Median':>8} {'Min':>6} {'Max':>6}\n")
    f.write("-"*55 + "\n")
    for _, row in t4.iterrows():
        f.write(f"{row['Cycle']:<12} {row['Mean']:>8.1f} {row['SD']:>8.1f} "
                f"{row['Median']:>8.1f} {row['Min']:>6.0f} {row['Max']:>6.0f}\n")
print(f"    Saved: table4_age_distribution")

print(f"\n    All tables saved to: {tables_dir}")
print(f"    Formats: .csv (for code) and .txt (formatted, paste into paper)")

print(f"\n{'='*60}")
print("EXPLORATION COMPLETE")



print(f"{'='*60}")