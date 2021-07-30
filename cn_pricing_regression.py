import numpy as np
from numpy import float64, random
import pandas as pd
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# load the csv & print columns' info 
df = pd.read_csv('cn_provider_pricing_dummy.csv')

print('rows x columns:', df.shape)
print('Columns info:', df.info())
print('Data highlights:', df.describe())

# Check for null values
print(df.isnull().sum() * 100 / df.shape[0])

# Outlier Analysis for numeric variables
fig, axs = plt.subplots(1, 4, figsize=(15, 6))
plt1 = sns.boxplot(df['Price'], ax=axs[0])
plt2 = sns.boxplot(df['CPU'], ax=axs[1])
plt3 = sns.boxplot(df['RAM'], ax=axs[2])
plt4 = sns.boxplot(df['STORAGE'], ax=axs[3])

plt.tight_layout()
plt.show()

# Visualize numeric variables
sns.pairplot(df)
plt.plot(color='green')
plt.show()

# Visualize categorical variables
plt.figure(figsize=(20, 28))
plt.subplot(5, 3, 1)
sns.boxplot(x='Cluster_management_fee', y='Price', data=df)
plt.subplot(5, 3, 2)
sns.boxplot(x='Regional redundancy', y='Price', data=df)
plt.subplot(5, 3, 3)
sns.boxplot(x='Auto-scaling', y='Price', data=df)
plt.subplot(5, 3, 4)
sns.boxplot(x='Vendor_lock-in', y='Price', data=df)
plt.subplot(5, 3, 5)
sns.boxplot(x='Payment_option', y='Price', data=df)
plt.subplot(5, 3, 6)
sns.boxplot(x='Term_Length', y='Price', data=df)
plt.subplot(5, 3, 7)
sns.boxplot(x='Instance_Type', y='Price', data=df)
plt.subplot(5, 3, 8)
sns.boxplot(x='Hybrid&multi-cloud_support', y='Price', data=df)
plt.subplot(5, 3, 9)
sns.boxplot(x='Pay_per_pod_usage', y='Price', data=df)
plt.subplot(5, 3, 10)
sns.boxplot(x='Built-in_authentication', y='Price', data=df)
plt.subplot(5, 3, 11)
sns.boxplot(x='self-recovery_features', y='Price', data=df)
plt.subplot(5, 3, 12)
sns.boxplot(x='automate_backup_tasks', y='Price', data=df)
plt.subplot(5, 3, 13)
sns.boxplot(x='Monitoring&logging', y='Price', data=df)
plt.subplot(5, 3, 14)
sns.boxplot(x='Versioning&upgrades', y='Price', data=df)
plt.show()


# # Visualize categorical features parallel
# plt.figure(figsize = (10, 5))
# sns.boxplot(x = 'furnishingstatus', y = 'price', hue = '', data=df)
# plt.show()

# List of variables to map

varlist = ['Cluster_management_fee', 'Regional redundancy' ]
