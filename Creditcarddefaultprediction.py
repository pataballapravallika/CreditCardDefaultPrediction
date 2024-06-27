#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np

# Load data

# Handle missing values

import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.

cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='nIeaqxvHSeexPAaxzxbdnJeFVHwSBKjP_jzkf450RjMq',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us-south.cloud-object-storage.appdomain.cloud')

bucket = 'creditcarddefaultprediction-donotdelete-pr-mc32gs3af4hd70'
object_key = 'UCI_Credit_Card.csv'

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_1 = pd.read_csv(body)
df_1.head(10)



# In[14]:


import os
import types
import pandas as pd
import numpy as np
from botocore.client import Config
import ibm_boto3

# Define __iter__ method for compatibility with pandas
def __iter__(self): return 0

# Access file from IBM Cloud Object Storage
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='nIeaqxvHSeexPAaxzxbdnJeFVHwSBKjP_jzkf450RjMq',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us-south.cloud-object-storage.appdomain.cloud')

bucket = 'creditcarddefaultprediction-donotdelete-pr-mc32gs3af4hd70'
object_key = 'UCI_Credit_Card.csv'

body = cos_client.get_object(Bucket=bucket, Key=object_key)['Body']
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType(__iter__, body)

# Load data into DataFrame
df_1 = pd.read_csv(body)
print(df_1.head(10))

# Handle missing values
df_1.fillna(method='ffill', inplace=True)

# Check for remaining missing values
print(df_1.isnull().sum())


# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot distributions
sns.histplot(df_1['LIMIT_BAL'], kde=True)
plt.show()

# Visualize correlations
corr_matrix = df_1.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()

# Relationships with target variable
sns.boxplot(x='default.payment.next.month', y='LIMIT_BAL', data=df_1)
plt.show()


# In[16]:


# Create new features
df_1['balance_limit_ratio'] = df_1['BILL_AMT1'] / df_1['LIMIT_BAL']

# Separate target variable before scaling
target = df_1['default.payment.next.month']
features = df_1.drop('default.payment.next.month', axis=1)

# Normalize data (excluding the target variable)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

# Combine scaled features and target variable
df_scaled = pd.concat([features_scaled, target.reset_index(drop=True)], axis=1)


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Ensure target variable is binary
df_scaled['default.payment.next.month'] = df_scaled['default.payment.next.month'].astype(int)

# Split data
X = df_scaled.drop('default.payment.next.month', axis=1)
y = df_scaled['default.payment.next.month']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict_proba(X_test)[:, 1]
print('ROC AUC Score:', roc_auc_score(y_test, y_pred))


# In[18]:


from sklearn.metrics import roc_curve

# Calculate KS statistic
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
ks_stat = max(tpr - fpr)
print('KS Statistic:', ks_stat)

# Plot K-S chart
plt.plot(thresholds, tpr - fpr)
plt.xlabel('Threshold')
plt.ylabel('K-S Statistic')
plt.title('K-S Chart')
plt.show()


# In[ ]:




