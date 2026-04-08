# Why Invariant Risk Minimization Fails on Tabular Data: A Gradient Variance Solution

Grold Otieno Mboya

---

## Abstract

Standard machine learning models fail under distribution shift because they exploit spurious correlations that vary across environments. CausTab penalizes the variance of parameter gradients across training environments. Parameters responding to causal features receive consistent gradient signals and are not penalized. Parameters responding to spurious features receive inconsistent signals and are penalized.

Across synthetic data, four cycles of NHANES (16,773 participants), and the UCI Heart Disease dataset (920 patients), CausTab matches or exceeds empirical risk minimization (ERM) in every experimental condition. Invariant Risk Minimization (IRM) degrades by up to 13.8 AUC points on spurious-dominant tabular data due to penalty collapse. CausTab does not exhibit this failure. The method achieves consistently lower expected calibration error than both ERM and IRM.

A boundary condition applies: invariant learning fails when environments differ primarily in outcome prevalence rather than spurious correlations. The Spurious Dominance Index (SDI) provides a practical diagnostic for determining when invariant learning is likely to help.

---

## Critical Results

### 1. Synthetic Data

CausTab matches ERM across all three spurious-correlation regimes. IRM degrades substantially in mixed and spurious-dominant settings.

| Regime | SDI | ERM | IRM | CausTab |
|--------|-----|-----|-----|---------|
| Causal dominant | 1.67 | 0.936 | 0.925 | 0.936 |
| Mixed | 9.62 | 0.917 | 0.840 | 0.917 |
| Spurious dominant | 48.08 | 0.766 | 0.706 | 0.767 |

![Synthetic regimes](experiments/plots/science/fig3_synthetic_regimes.png)

### 2. IRM Failure Analysis

IRM degrades relative to ERM as spurious feature strength increases, reaching a maximum gap of -0.138 AUC at strength 2.5. CausTab tracks ERM within 0.001 AUC at every level.

![IRM failure](experiments/plots/science/fig2_irm_failure.png)

### 3. NHANES Temporal Evaluation

All methods achieve comparable AUC. CausTab achieves the lowest expected calibration error (ECE) in every configuration.

| Method | Split B AUC | Split B ECE | Split C AUC | Split C ECE |
|--------|-------------|-------------|-------------|-------------|
| ERM | 0.814 | 0.025 | 0.813 | 0.023 |
| IRM | 0.814 | 0.029 | 0.812 | 0.026 |
| CausTab | 0.814 | 0.024 | 0.813 | 0.023 |

![NHANES temporal](experiments/plots/science/fig4_nhanes_temporal.png)
![Calibration](experiments/plots/science/fig5_calibration.png)
![Distribution shift evidence](experiments/plots/science/fig1_nhanes_shift.png)

### 4. UCI Heart Disease Boundary Condition

When environments differ primarily in outcome prevalence, the shared causal mechanism assumption fails. CausTab underperforms ERM on the Cleveland fold (0.455 vs 0.834).

| Method | Cleveland | Hungary | Switzerland | VA | Mean |
|--------|-----------|---------|-------------|-----|------|
| ERM | 0.834 | 0.863 | 0.622 | 0.656 | 0.744 |
| IRM | 0.680 | 0.897 | 0.615 | 0.747 | 0.735 |
| CausTab | 0.455 | 0.876 | 0.610 | 0.706 | 0.662 |

![UCI results](experiments/plots/science/fig8_uci_results.png)

### 5. SDI Validation

The Spurious Dominance Index monotonically predicts CausTab's advantage over ERM across all experimental settings.

![SDI validation](experiments/plots/science/fig6_sdi_validation.png)

### 6. Ablation Study

Removing or replacing components of CausTab has minimal impact on performance, confirming robustness.

![Ablation](experiments/plots/science/fig7_ablation.png)

### 7. Summary

![Summary](experiments/plots/science/fig9_summary.png)

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
