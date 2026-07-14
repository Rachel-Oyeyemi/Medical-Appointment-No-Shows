# Model Evaluation

Appointments are split chronologically into 70% training, 15% validation, and 15% test partitions. Thresholds are selected only on validation data using illustrative costs of 5 for a missed no-show and 1 for unnecessary outreach.

Primary metric: PR-AUC. Supporting metrics: accuracy, precision, recall, F1, ROC-AUC, balanced accuracy, MCC, specificity, outreach rate, and confusion matrix.

The demonstration threshold is intentionally recall-oriented. A real clinic must choose thresholds from actual capacity, costs, patient preferences, and equity analysis.
