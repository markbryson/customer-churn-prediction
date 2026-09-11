import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#LOADING DATASETS

file_path = "Customer_Churn_Data_Large.xlsx"

demographics = pd.read_excel(
    file_path,
    sheet_name='Customer_Demographics'
)

transactions = pd.read_excel(
    file_path,
    sheet_name='Transaction_History'
)

service = pd.read_excel(
    file_path,
    sheet_name='Customer_Service'
)

online = pd.read_excel(
    file_path,
    sheet_name='Online_Activity'
)

churn = pd.read_excel(
    file_path,
    sheet_name='Churn_Status'
)


#REMOVING DUPLICATES

online = online.drop_duplicates(
    subset='CustomerID',
    keep='last'
)

churn = churn.drop_duplicates(
    subset='CustomerID',
    keep='last'
)


#FEATURE ENGINEERING

#Transaction features
transaction_features = transactions.groupby(
    'CustomerID'
).agg({
    'AmountSpent': ['sum', 'mean', 'count'],
    'ProductCategory': 'nunique'
})

transaction_features.columns = [
    'TotalSpend',
    'AverageSpend',
    'TransactionCount',
    'UniqueProducts'
]

transaction_features = transaction_features.reset_index()

#Service features
service_features = service.groupby(
    'CustomerID'
).agg({
    'InteractionID': 'count'
})

service_features.columns = ['ServiceInteractions']

service_features = service_features.reset_index()


#MERGING DATASETS

df = demographics.merge(
    transaction_features,
    on='CustomerID',
    how='left'
)

df = df.merge(
    service_features,
    on='CustomerID',
    how='left'
)

df = df.merge(
    online,
    on='CustomerID',
    how='left'
)

df = df.merge(
    churn,
    on='CustomerID',
    how='left'
)


#HANDLING MISSING VALUES

df['ServiceInteractions'] = df[
    'ServiceInteractions'
].fillna(0)

numeric_fill_cols = [
    'TotalSpend',
    'AverageSpend',
    'TransactionCount',
    'UniqueProducts',
    'ServiceInteractions',
    'LoginFrequency'
]

for col in numeric_fill_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)


#CHECKING MISSING VALUES

print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum())


#DESCRIPTIVE STATISTICS


print("\nDESCRIPTIVE STATISTICS")
print(df.describe())


#VISUALISATIONS

#Churn distribution
plt.figure(figsize=(6,4))

sns.countplot(
    x='ChurnStatus',
    data=df
)

plt.title('Customer Churn Distribution')
plt.xlabel('Churn Status')
plt.ylabel('Number of Customers')

plt.show()

#Age distribution
plt.figure(figsize=(6,4))

sns.histplot(
    df['Age'],
    bins=20,
    kde=True
)

plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.show()

#Total spend distribution
plt.figure(figsize=(6,4))

sns.histplot(
    df['TotalSpend'],
    bins=20,
    kde=True
)

plt.title('Total Spend Distribution')
plt.xlabel('Total Spend')
plt.ylabel('Frequency')

plt.show()

#Login frequency distribution
plt.figure(figsize=(6,4))

sns.histplot(
    df['LoginFrequency'],
    bins=20,
    kde=True
)

plt.title('Login Frequency Distribution')
plt.xlabel('Login Frequency')
plt.ylabel('Frequency')

plt.show()

#Boxplot - TotalSpend
plt.figure(figsize=(6,4))

sns.boxplot(
    x=df['TotalSpend']
)

plt.title('Boxplot of Total Spend')

plt.show()

#Boxplot - LoginFrequency
plt.figure(figsize=(6,4))

sns.boxplot(
    x=df['LoginFrequency']
)

plt.title('Boxplot of Login Frequency')

plt.show()

#Boxplot - ServiceInteractions
plt.figure(figsize=(6,4))

sns.boxplot(
    x=df['ServiceInteractions']
)

plt.title('Boxplot of Service Interactions')

plt.show()

#Correlation heatmap
numeric_df = df.select_dtypes(
    include=['int64', 'float64']
)

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(12,8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title('Correlation Heatmap')

plt.show()


#MACHINE LEARNING

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

#Creating ML copy
ml_df = df.copy()


#ENCODING CATEGORICAL DATA

categorical_columns = [
    'Gender',
    'MaritalStatus',
    'IncomeLevel',
    'ServiceUsage'
]

for column in categorical_columns:

    ml_df[column] = ml_df[column].fillna(
        'Unknown'
    )

    encoder = LabelEncoder()

    ml_df[column] = encoder.fit_transform(
        ml_df[column]
    )


#HANDLING TARGET VARIABLE

if ml_df['ChurnStatus'].dtype == object:

    churn_encoder = LabelEncoder()

    ml_df['ChurnStatus'] = churn_encoder.fit_transform(
        ml_df['ChurnStatus']
    )


#DATE FEATURE ENGINEERING

ml_df['LastLoginDate'] = pd.to_datetime(
    ml_df['LastLoginDate'],
    errors='coerce'
)

reference_date = ml_df[
    'LastLoginDate'
].max()

ml_df['DaysSinceLogin'] = (
    reference_date - ml_df['LastLoginDate']
).dt.days

ml_df['DaysSinceLogin'] = ml_df[
    'DaysSinceLogin'
].fillna(0)

ml_df.drop(
    'LastLoginDate',
    axis=1,
    inplace=True
)


#FEATURE SCALING

scaler = StandardScaler()

numerical_columns = [
    'Age',
    'TotalSpend',
    'AverageSpend',
    'TransactionCount',
    'UniqueProducts',
    'ServiceInteractions',
    'LoginFrequency',
    'DaysSinceLogin'
]

ml_df[numerical_columns] = scaler.fit_transform(
    ml_df[numerical_columns]
)


#FEATURES & TARGET

X = ml_df.drop(
    ['CustomerID', 'ChurnStatus'],
    axis=1
)

y = ml_df['ChurnStatus']


#TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTRAIN TEST SPLIT COMPLETE")

print(
    "Training Set Shape:",
    X_train.shape
)

print(
    "Testing Set Shape:",
    X_test.shape
)


#HYPERPARAMETER TUNING

param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}

base_model = LogisticRegression(
    max_iter=3000,
    class_weight='balanced'
)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(
    X_train,
    y_train
)

print("\nBEST PARAMETERS:")
print(grid_search.best_params_)

#Best model
model = grid_search.best_estimator_


#MODEL TRAINING


model.fit(
    X_train,
    y_train
)

#Predictions
y_pred = model.predict(X_test)

#Probability predictions
y_prob = model.predict_proba(X_test)[:,1]


#MODEL EVALUATION

print("\nMODEL EVALUATION")

#Accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(accuracy)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    y_pred
))

#Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred
))

#ROC-AUC
roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("\nROC-AUC Score:")
print(roc_auc)


#CROSS VALIDATION

kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=kfold,
    scoring='f1'
)

print("\nCross Validation F1 Scores:")
print(cv_scores)

print("\nAverage F1 Score:")
print(cv_scores.mean())


#ROC CURVE

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(6,4))

plt.plot(
    fpr,
    tpr,
    label='Logistic Regression'
)

plt.plot(
    [0,1],
    [0,1],
    linestyle='--'
)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend()

plt.show()

#FEATURE IMPORTANCE

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

feature_importance['AbsoluteCoefficient'] = abs(
    feature_importance['Coefficient']
)

feature_importance = feature_importance.sort_values(
    by='AbsoluteCoefficient',
    ascending=False
)

print("\nTOP IMPORTANT FEATURES")
print(feature_importance[
    ['Feature', 'Coefficient']
].head(10))


#FEATURE IMPORTANCE PLOT

plt.figure(figsize=(10,6))

sns.barplot(
    x='Coefficient',
    y='Feature',
    data=feature_importance.head(10)
)

plt.title('Top 10 Important Features')

plt.xlabel('Coefficient Value')
plt.ylabel('Features')

plt.show()