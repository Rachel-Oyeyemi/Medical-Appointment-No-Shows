# Exploratory Data Analysis Report

The widely reproduced raw profile contains 110,527 Brazilian appointments and 14 columns. `No-show = Yes` is the positive class and represents roughly 20.2% of records. Published reproductions report no missing values and no exact duplicate rows.

## Quality Issues
- At least one invalid negative age
- `Handcap` values above one
- Spelling inconsistencies in source column names
- Potential negative waiting times after date parsing
- Identifier and repeat-patient leakage risk
- SMS status is observational and may be confounded by lead time

## Analysis
The project examines class balance, age, waiting days, SMS status, age groups, neighbourhoods, chronic-condition indicators, correlations, and outliers. Accuracy is insufficient because attendance is the majority outcome.
