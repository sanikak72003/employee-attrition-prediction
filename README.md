# Employee Attrition Prediction

## Project Overview

This project uses machine learning to predict whether an employee is likely to leave an organization based on employee-related information.

The objective is to explore employee attrition patterns, perform data analysis, train classification models, compare their performance, and interpret model predictions.

## Dataset

- IBM HR Employee Attrition dataset
- 1,470 employee records
- Target variable: `Attrition`
  - `0`: Stayed
  - `1`: Left

## Project Workflow

1. Data loading and inspection
2. Data cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
   - Univariate analysis
   - Bivariate analysis
   - Multivariate analysis
   - Correlation and outlier analysis
4. Feature encoding and scaling
5. Train-test split with stratification
6. Classification model training
7. Model comparison and hyperparameter tuning
8. Cross-validation
9. Model interpretation using feature importance and SHAP
10. Model serialization using Joblib

## Models Implemented

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

## Final Model: XGBoost

The selected XGBoost model achieved the following test-set results:

| Metric | Score |
|---|---:|
| Accuracy | 80% |
| Precision (Attrition) | 43% |
| Recall (Attrition) | 68% |
| F1-score (Attrition) | 52% |
| ROC-AUC | 0.798 |
| PR-AUC | 0.511 |

The model identifies a substantial proportion of employees who leave, although its predictions also include false positives.

**Evaluation note:** The test set was used during manual hyperparameter experimentation. Therefore, these results should be treated as exploratory rather than as an unbiased final performance estimate.

## Model Interpretability

Feature importance and SHAP were used to understand which features influenced the model's predictions.

The most influential features by mean absolute SHAP value included:

- OverTime_Yes
- StockOptionLevel
- Age
- MonthlyIncome
- NumCompaniesWorked

SHAP explains the model's learned associations; it does not establish causal relationships.

## Project Structure

```text
Employee Attrition Prediction/
│
├── datasets/
│   └── employeeattrition.csv
│
├── notebooks/
│   └── employee_attrition.ipynb
│
├── models/
│   └── employee_attrition_model.pkl
│
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- SHAP
- Joblib
- Jupyter Notebook

## Installation

Clone the repository:

```bash
git clone <repository-url>
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook using Jupyter Notebook and run the analysis.

## Future Improvements

- More rigorous hyperparameter validation
- Threshold optimization based on business requirements
- Improved model generalization
- Deployment through an interactive prediction interface