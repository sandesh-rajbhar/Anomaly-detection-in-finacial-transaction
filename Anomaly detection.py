#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_excel(r"C:\Users\sande\OneDrive\Desktop\50K.xlsx")

# Display the first few rows of the dataset
print(data.head())

# Summary statistics
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Distribution of numerical features
data.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

# Distribution of categorical features
plt.figure(figsize=(8, 6))
sns.countplot(x='type', data=data)
plt.title('Distribution of Transaction Types')
plt.xlabel('Transaction Type')
plt.ylabel('Count')
plt.show()

# Transaction patterns over time
plt.figure(figsize=(12, 6))
sns.lineplot(x='step', y='amount', data=data)
plt.title('Transaction Amount Over Time')
plt.xlabel('Step')
plt.ylabel('Amount')
plt.show()

# Fraud analysis
fraud_percentage = (data['isFraud'].sum() / len(data)) * 100
print("Percentage of fraudulent transactions:", fraud_percentage)

# Correlation analysis
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Anomaly detection visualization
plt.figure(figsize=(10, 6))
plt.scatter(data['amount'], data['newbalanceDest'], c=data['isFlaggedFraud'], cmap='coolwarm')
plt.xlabel('Amount')
plt.ylabel('New Balance Destination')
plt.title('Anomaly Detection Visualization')
plt.colorbar(label='isFlaggedFraud')
plt.show()


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# Load the transaction data
data = pd.read_csv(r"C:\Users\sande\OneDrive\Desktop\trimmed_dataset.csv")  # Assuming the data is stored in a CSV file named "transaction_data.csv"

# Convert categorical variables to numerical labels
label_encoder = LabelEncoder()
data['type'] = label_encoder.fit_transform(data['type'])
data['nameOrig'] = label_encoder.fit_transform(data['nameOrig'])
data['nameDest'] = label_encoder.fit_transform(data['nameDest'])

# Select relevant features for anomaly detection
features = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 
            'nameDest', 'oldbalanceDest', 'newbalanceDest']

# Initialize and fit the Isolation Forest model
model = IsolationForest(contamination=0.01)  # Assuming a contamination rate of 1%
model.fit(data[features])

# Predict anomalies
data['anomaly'] = model.predict(data[features])

# Anomalies will be marked as -1, normal instances will be marked as 1
anomalies = data[data['anomaly'] == -1]

# Plotting the distribution of normal and anomalous transactions
plt.figure(figsize=(10, 6))

plt.scatter(data.index, data['amount'], label='Normal', c='blue', alpha=0.5)
plt.scatter(anomalies.index, anomalies['amount'], label='Anomaly', c='red')

plt.title('Anomalies Detected by Isolation Forest')
plt.xlabel('Transaction Index')
plt.ylabel('Transaction Amount')
plt.legend()

plt.show()

# Print detected anomalies
print("Detected anomalies:")
print(anomalies)


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load the transaction data
data = pd.read_csv(r"C:\Users\sande\OneDrive\Desktop\trimmed_dataset.csv")  # Assuming the data is stored in a CSV file named "trimmed_dataset.csv"

# Normalize numerical features
scaler = StandardScaler()
numerical_features = ['amount', 'newbalanceDest']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Select relevant features for anomaly detection
features = ['amount', 'newbalanceDest']

# Initialize and fit the Isolation Forest model
model = IsolationForest(contamination=0.01)  # Assuming a contamination rate of 1%
model.fit(data[features])

# Predict anomalies
data['anomaly'] = model.predict(data[features])

# Anomalies will be marked as -1, normal instances will be marked as 1
anomalies = data[data['anomaly'] == -1]

# Plotting the distribution of normal and anomalous transactions
plt.figure(figsize=(10, 6))
plt.scatter(data['amount'], data['newbalanceDest'], c='blue', alpha=0.5, label='Normal')
plt.scatter(anomalies['amount'], anomalies['newbalanceDest'], c='red', alpha=0.5, label='Anomaly')

plt.title('Anomalies Detected by Isolation Forest')
plt.xlabel('Transaction Amount')
plt.ylabel('New Balance Destination')
plt.legend()
plt.show()

# Print detected anomalies
print("Detected anomalies:")
print(anomalies)


# In[ ]:




