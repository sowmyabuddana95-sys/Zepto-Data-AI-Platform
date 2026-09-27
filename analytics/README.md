
# Module 2 — Titanic Analytics and Machine Learning

## Overview

This module performs exploratory data analysis, preprocessing, classification,
hyperparameter tuning, class-imbalance analysis, and multivariate linear
regression using the classic Titanic dataset.

The dataset was loaded once using `seaborn.load_dataset("titanic")` and saved
as `titanic.csv` for offline reproducibility. The same cleaned dataset was
used throughout the analysis and modeling workflow.

## Dataset

Original dataset:
- Rows: 891
- Columns: 15

Missing values in the original dataset:
- age: 177 missing (19.87%)
- embarked: 2 missing (0.22%)
- deck: 688 missing (77.22%)
- embark_town: 2 missing (0.22%)

Cleaning strategy:
- `age` (19.87% missing): median imputation using median = 28.0
- `embarked` (0.22% missing): dropped rows
- `embark_town` (0.22% missing): dropped rows
- `deck` (77.22% missing): dropped because of very high missingness

Final cleaned dataset:
- Rows: 889
- Columns: 14
- Remaining missing values: none

## Exploratory Data Analysis

### Age and Fare

IQR analysis:
- Age Q1: 22.0
- Age Q3: 35.0
- Age IQR: 13.0
- Age outliers: 65
- Fare Q1: 7.8958
- Fare Q3: 31.0
- Fare IQR: 23.1042
- Fare outliers: 114

Fare statistics:
- Mean: 32.0967
- Median: 14.4542
- Mode: 8.05
- Skewness: 4.8014

Since mean > median > mode and skewness is strongly positive, fare is
right-skewed.

### Survival Analysis

Survival rate by sex:
- Female: 74.04%
- Male: 18.89%

Survival rate by passenger class:
- First class: 62.62%
- Second class: 47.28%
- Third class: 24.24%

Survival by sex and class showed substantial differences across groups,
with female passengers having higher survival rates within each class.

### Correlation Analysis

The required six-column correlation matrix used:
- survived
- pclass
- age
- sibsp
- parch
- fare

Strongest absolute off-diagonal correlations:
- pclass vs fare: -0.5482
- sibsp vs parch: 0.4145

### Multivariate Charts

Four multivariate charts were created:
1. Survival rate by passenger class and sex
2. Passenger distribution by class and sex
3. Fare distribution by survival status
4. Age distribution by sex and survival status

The charts showed clear differences in survival patterns across sex and
passenger class, differences in passenger composition across classes, higher
fare distributions among survivors, and age-distribution differences across
sex and survival groups.

## Standardization

Exploratory z-score standardization was applied to age and fare.

Before standardization:
- Age mean: 29.3152
- Age standard deviation: 12.9849
- Fare mean: 32.0967
- Fare standard deviation: 49.6975

After standardization, both variables had means approximately equal to zero
and standard deviations approximately equal to one.

## Classification

A stratified 80/20 train-test split was used.

Class distribution:
- Overall: 61.75% non-survived, 38.25% survived
- Training: 61.74% non-survived, 38.26% survived
- Test: 61.80% non-survived, 38.20% survived

Features used for classification:
- pclass
- sex
- age
- sibsp
- parch
- fare
- embarked

Preprocessing:
- Numeric features: median imputation + StandardScaler
- Categorical features: most-frequent imputation + one-hot encoding
- Preprocessing was fitted only on the training data.

### Classification Results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8090 | 0.7833 | 0.6912 | 0.7344 | 0.8610 |
| Decision Tree | 0.7640 | 0.7600 | 0.5588 | 0.6441 | 0.8374 |
| Random Forest | 0.8090 | 0.7656 | 0.7206 | 0.7424 | 0.8196 |
| Tuned Random Forest | 0.8315 | 0.8654 | 0.6618 | 0.7500 | 0.8389 |

The tuned Random Forest achieved 0.8315 accuracy, 0.8654 precision, and
0.7500 F1. Logistic Regression achieved the highest ROC AUC at 0.8610.
The tuned Random Forest was selected as the final saved classification
pipeline based on its accuracy, precision, and F1 results, while its lower
recall compared with the original Random Forest was noted.

## Class Imbalance

The following Logistic Regression approaches were compared:

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| Baseline | 0.7833 | 0.6912 | 0.7344 |
| Class Weight Balanced | 0.7183 | 0.7500 | 0.7338 |
| SMOTE | 0.7353 | 0.7353 | 0.7353 |

Class weighting increased recall while reducing precision. SMOTE produced
balanced precision and recall of 0.7353. The F1 scores remained very close
across all three approaches.

SMOTE was applied only to the training data.

## Random Forest Grid Search

GridSearchCV searched:
- n_estimators: 100, 200
- max_depth: None, 5, 10
- max_features: sqrt, log2

Best parameters:
- n_estimators: 200
- max_depth: 5
- max_features: sqrt

Best cross-validation F1: 0.7408
OOB score: 0.8214

## Fare Regression

A multivariate Linear Regression model was used to predict fare from:
- pclass
- sex
- age
- sibsp
- parch
- embarked

Results:
- MAE: 21.1386
- RMSE: 41.7465
- R²: 0.3468
- Adjusted R²: 0.3118

The residual plot showed evidence of non-constant variance. Residuals had
a wider spread at higher predicted fare values, with several large positive
residuals, suggesting heteroscedasticity.

## Final Model

The complete fitted tuned Random Forest pipeline was saved as:

`best_titanic_pipeline.joblib`

The saved pipeline contains both:
1. preprocessing
2. Random Forest classifier

The pipeline was reloaded successfully and tested using raw passenger input.

Example:
- Actual survival: 0
- Predicted survival: 0
- Predicted survival probability: 0.1684

## Files

- `01_eda.ipynb` — complete EDA and modeling workflow
- `titanic.csv` — saved Titanic dataset for offline reproducibility
- `best_titanic_pipeline.joblib` — complete fitted classification pipeline
- `README.md` — Module 2 documentation

## Completion Status

Module 2 — Analytics: Complete.
