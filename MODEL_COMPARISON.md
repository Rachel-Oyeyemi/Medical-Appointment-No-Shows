# Model Comparison

## Baseline
Class-weighted Logistic Regression with standardized numeric variables and one-hot categories. Transparent, fast, and explainable.

## Advanced
Regularized XGBoost with positive-class weighting. Captures nonlinear thresholds and interactions.

## Demonstration Results
| Model | Test PR-AUC | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.468 | 0.294 | 0.861 | 0.438 | 0.765 |
| XGBoost | 0.454 | 0.315 | 0.804 | 0.453 | 0.758 |

XGBoost wins validation PR-AUC by a very small margin; Logistic Regression ranks better and has higher recall on the synthetic test set. Preserve both as champion/challenger candidates and rerun on official data.
