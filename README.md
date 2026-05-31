# Insurance Loyalty Prediction

## Overview
Predicting insurance policy renewal on a highly imbalanced dataset (5% vs 95% 
class split) using XGBoost classifier with Grid Search hyperparameter tuning.

## Dataset
- 53,236 training records, 26,617 test records
- Target: renewal (1 = renewed, 0 = not renewed)
- Class imbalance: ~5% non-renewal vs ~95% renewal
- Features include: premium payment behaviour, late payment counts, 
  income, age, underwriting score, sourcing channel

## Approach

### Missing Value Imputation
- Delay count nulls filled with -1 (new customer assumption)
- Underwriting score (1,976 missing values) imputed using XGBoost 
  regression — MAE: 0.397

### Feature Analysis
- Mutual information scoring to rank feature importance
- Top features: premium payment percentage, late payment counts, age, income

### Modelling
- XGBoost classifier with StratifiedKFold cross-validation
- Grid Search hyperparameter tuning
- Oversampling applied to handle class imbalance
- Local Outlier Factor tested and rejected (F1 dropped from 0.398 to 0.311)

### Best Model Parameters
