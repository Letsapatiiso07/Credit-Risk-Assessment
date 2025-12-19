# Credit Risk Assessment using Machine Learning

A machine learning project to predict loan default risk using borrower financial and demographic data.

**Dataset**: [Credit Risk Dataset by laotse (Kaggle)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)  
**Target**: Predict `loan_status` (1 = default, 0 = no default)  
**Best Model**: XGBoost  
**Performance**: **AUC-ROC: ~0.945** (excellent!)

## Project Overview

This project demonstrates an end-to-end machine learning workflow for credit risk modeling in fintech:

- Exploratory Data Analysis (EDA)
- Feature Engineering (e.g., debt-to-income ratio)
- Handling missing data and outliers
- Categorical encoding and scaling
- Class imbalance handling
- Model training and evaluation (Logistic Regression, KNN, XGBoost)
- Model interpretation via feature importance

### Key Insights
- Highest risk indicators: High loan grade (D-G), high interest rate, renting vs owning home
- Engineered features like debt-to-income significantly improve prediction

## Requirements

See `requirements.txt`

To install:
```bash
pip install -r requirements.txt