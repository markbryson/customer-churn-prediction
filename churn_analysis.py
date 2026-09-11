import pandas as pd

#Loading the workbook
file_path = "Customer_Churn_Data_Large.xlsx"

#ReadIing datasets
demographics = pd.read_excel(file_path, sheet_name='Customer_Demographics')
transactions = pd.read_excel(file_path, sheet_name='Transaction_History')
service = pd.read_excel(file_path, sheet_name='Customer_Service')
online = pd.read_excel(file_path, sheet_name='Online_Activity')
churn = pd.read_excel(file_path, sheet_name='Churn_Status')

# Ensure there is only one record per customer for one-to-one merges
online = online.drop_duplicates(subset='CustomerID', keep='last')
churn = churn.drop_duplicates(subset='CustomerID', keep='last')


#TRANSACTION FEATURES
transaction_features = transactions.groupby('CustomerID').agg({
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


#SERVICE FEATURES
service_features = service.groupby('CustomerID').agg({
    'InteractionID': 'count'
})

service_features.columns = ['ServiceInteractions']

service_features = service_features.reset_index()


#MERGING THE DATASETS
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

df['ServiceInteractions'] = df['ServiceInteractions'].fillna(0)

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

print("\n MISSING VALUES AFTER CLEANING")

print(df.isnull().sum())

#DESCRIPTIVE STATISTICS

print("\n DESCRIPTIVE STATISTICS")
print(df.describe())


#CHURN DISTRIBUTION

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))

sns.countplot(
    x='ChurnStatus',
    data=df
)

plt.title('Customer Churn Distribution')

plt.xlabel('Churn Status')

plt.ylabel('Number of Customers')

plt.show()


#HISTOGRAMS

#AGE DISTRIBUTION
plt.figure(figsize=(6,4))

sns.histplot(df['Age'], bins=20, kde=True)

plt.title('Age Distribution')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()


#TOTAL SPEND DISTRIBUTION
plt.figure(figsize=(6,4))

sns.histplot(df['TotalSpend'], bins=20, kde=True)

plt.title('Total Spend Distribution')

plt.xlabel('Total Spend')

plt.ylabel('Frequency')

plt.show()


#LOGIN FREQUENCY DISTRIBUTION
plt.figure(figsize=(6,4))

sns.histplot(df['LoginFrequency'], bins=20, kde=True)

plt.title('Login Frequency Distribution')

plt.xlabel('Login Frequency')

plt.ylabel('Frequency')

plt.show()


#BOXPLOTS

#TOTAL SPEND BOXPLOT
plt.figure(figsize=(6,4))

sns.boxplot(x=df['TotalSpend'])

plt.title('Boxplot of Total Spend')

plt.show()


#LOGIN FREQUENCY BOXPLOT
plt.figure(figsize=(6,4))

sns.boxplot(x=df['LoginFrequency'])

plt.title('Boxplot of Login Frequency')

plt.show()


#SERVICE INTERACTIONS BOXPLOT
plt.figure(figsize=(6,4))

sns.boxplot(x=df['ServiceInteractions'])

plt.title('Boxplot of Service Interactions')

plt.show()


#CORRELATION HEATMAP
#Select only numerical columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

#Correlation matrix
correlation_matrix = numeric_df.corr()

#Plot heatmap
plt.figure(figsize=(12,8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title('Correlation Heatmap')

plt.show()


#MACHINE LEARNING PREPARATION

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


#Create copy
ml_df = df.copy()


#Encoding categorical columns
categorical_columns = [
    'Gender',
    'MaritalStatus',
    'IncomeLevel',
    'ServiceUsage'
]

for column in categorical_columns:
    if column in ml_df.columns:
        ml_df[column] = ml_df[column].fillna('Unknown')
        ml_df[column] = LabelEncoder().fit_transform(ml_df[column])


#Prepare churn label for modeling
if 'ChurnStatus' in ml_df.columns:
    if ml_df['ChurnStatus'].dtype == object:
        ml_df['ChurnStatus'] = LabelEncoder().fit_transform(
            ml_df['ChurnStatus'].fillna('Unknown')
        )
    ml_df = ml_df.dropna(subset=['ChurnStatus'])


#Converting LastLoginDate into numeric
if 'LastLoginDate' in ml_df.columns:
    ml_df['LastLoginDate'] = pd.to_datetime(
        ml_df['LastLoginDate'],
        errors='coerce'
    )
    reference_date = ml_df['LastLoginDate'].max()
    if pd.isna(reference_date):
        reference_date = pd.Timestamp.today()
    ml_df['DaysSinceLogin'] = (
        reference_date - ml_df['LastLoginDate']
    ).dt.days.fillna(0)
    ml_df.drop('LastLoginDate', axis=1, inplace=True)
else:
    ml_df['DaysSinceLogin'] = 0


from sklearn.preprocessing import StandardScaler

#Scale numerical features
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

existing_numerical = [col for col in numerical_columns if col in ml_df.columns]
for col in existing_numerical:
    ml_df[col] = ml_df[col].fillna(0)

ml_df[existing_numerical] = scaler.fit_transform(ml_df[existing_numerical])


#Features and target
X = ml_df.drop('CustomerID','ChurnStatus', axis=1)

y = ml_df['ChurnStatus']


#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTRAIN TEST SPLIT COMPLETE")

print("Training Set Shape:", X_train.shape)

print("Testing Set Shape:", X_test.shape)


#LOGISTIC REGRESSION MODEL

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


#Create model
model = LogisticRegression(
    max_iter=2000,
    class_weight='balanced'
)

#Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)


#EVALUATION

print("\n MODEL EVALUATION")

#Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

#Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))