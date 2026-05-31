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
booster: gbtree
max_depth: 6
learning_rate: 0.1
n_estimators: 500
gamma: 7
colsample_bytree: 0.9
scale_pos_weight: 0.2
reg_alpha: 0.04
reg_lambda: 0.2

## Results
- Best F1-score: **0.398** (minority class, 5-fold cross-validation)
- LOF outlier removal tested and rejected — reduced F1 to 0.311
- Final predictions: 24,303 renewals, 2,314 non-renewals on test set

## Key Decisions
- Used XGBoost regression (not mean/median) for underwriting score 
  imputation due to non-normalised data and categorical features
- Applied scale_pos_weight to handle class imbalance
- Ensured all preprocessing applied consistently to train and test 
  sets to prevent data leakage

## Tech Stack
- Python (Pandas, NumPy)
- XGBoost
- Scikit-learn (GridSearchCV, StratifiedKFold, LOF, mutual_info_classif)
- Matplotlib
