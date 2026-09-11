# Customer Churn Prediction

A machine learning project focused on analyzing customer churn and building a predictive classification model using Python and scikit-learn.

This project was completed as part of the Lloyds Banking Group Data Science Job Simulation on Forage.

## Project Overview

Customer churn prediction helps organizations identify customers who may be at risk of leaving. This project combines customer demographics, transaction history, customer service interactions, and online activity to engineer predictive features and build a machine learning model for churn classification.

The workflow covers data preparation, exploratory data analysis, feature engineering, preprocessing, model training, hyperparameter tuning, evaluation, and feature analysis.

## Dataset

The analysis uses a multi-sheet Excel dataset containing 1,000 customer records across:

* Customer demographics
* Transaction history
* Customer service interactions
* Online activity
* Churn status

The original Excel dataset is intentionally excluded from this repository because it is not required to reproduce the project code structure and should not be publicly redistributed.

## Data Preparation

The project performs several preprocessing steps:

* Loads data from multiple Excel worksheets using pandas
* Removes duplicate customer records where necessary
* Aggregates transaction-level data into customer-level features
* Calculates total spend, average spend, transaction count, and unique products
* Counts customer service interactions
* Merges the datasets using `CustomerID`
* Handles missing numerical values
* Encodes categorical variables using LabelEncoder
* Converts login dates into a `DaysSinceLogin` feature
* Standardizes numerical features using StandardScaler

### Engineered Features

* `TotalSpend`
* `AverageSpend`
* `TransactionCount`
* `UniqueProducts`
* `ServiceInteractions`
* `DaysSinceLogin`

## Exploratory Data Analysis

The analysis includes:

* Churn distribution
* Age distribution
* Total spending distribution
* Login frequency distribution
* Boxplots for spending, login frequency, and service interactions
* Numerical correlation heatmap

After cleaning, the dataset contained no missing values.

The dataset contained:

* 1,000 customers
* 204 churned customers
* 20.4% churn rate

## Machine Learning

The project uses Logistic Regression as the classification model.

The machine learning workflow includes:

1. Feature preparation
2. Categorical encoding
3. Numerical feature scaling
4. Stratified train-test split
5. Logistic Regression
6. GridSearchCV hyperparameter tuning
7. Five-fold stratified cross-validation
8. Model evaluation
9. ROC-AUC analysis
10. Feature coefficient analysis

### Hyperparameter Tuning

GridSearchCV evaluated different values of:

* `C`: 0.01, 0.1, 1, 10
* `solver`: `liblinear`, `lbfgs`

The best parameters were:

```text
C = 1
solver = liblinear
```

## Model Results

The final model was evaluated on a 20% test set containing 200 customers.

| Metric            | Result |
| ----------------- | -----: |
| Accuracy          |  52.5% |
| ROC-AUC           | 0.5502 |
| Average 5-Fold F1 | 0.2960 |
| Churn Recall      |    61% |

### Confusion Matrix

```text
[[80, 79],
 [16, 25]]
```

The model achieved 61% recall for the churn class, meaning it identified a majority of the actual churners in the test set. However, the relatively low ROC-AUC and F1 score show that the current model has limited predictive performance.

## Feature Analysis

The largest model coefficients included:

| Feature          | Coefficient |
| ---------------- | ----------: |
| TotalSpend       |     -0.3813 |
| AverageSpend     |      0.3010 |
| TransactionCount |      0.2387 |
| Gender           |      0.1984 |
| LoginFrequency   |     -0.1490 |

The coefficients represent model associations and should not be interpreted as causal relationships.

## Project Structure

```text
customer-churn-prediction/
├── churn_analysis.py
├── task2_model.py
├── README.md
└── .gitignore
```

## Technologies

* Python
* pandas
* NumPy
* scikit-learn
* Matplotlib
* Seaborn
* Excel
* VS Code

## Key Learning Outcomes

* Data cleaning and preprocessing
* Exploratory data analysis
* Feature engineering
* Data visualization
* Classification modeling
* Handling class imbalance
* Feature scaling
* Hyperparameter tuning
* Cross-validation
* ROC-AUC evaluation
* Model interpretation

## Limitations

The current model has limited predictive performance.

Potential improvements include:

* Testing Random Forest and Gradient Boosting models
* More advanced feature engineering
* Alternative encoding approaches
* More extensive hyperparameter optimization
* Additional class-imbalance techniques
* Comparing multiple classification algorithms

## Lloyds Banking Group Data Science Job Simulation

This project was developed as part of the Lloyds Banking Group Data Science Job Simulation on Forage.

The simulation provided practical experience in customer churn analysis, predictive modeling, model evaluation, and communicating data-driven insights.

## Author

**Mark Bryson Mutuma**

Data Science Student | Machine Learning | Applied AI
