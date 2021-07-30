from sklearn import linear_model

import multi_linear_reg
import numpy as np
import pandas as pd
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings('ignore')  # it is used for some minor warnings in seaborn

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
sns.boxplot(x='Regional_redundancy', y='Price', data=df)
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

# Visualize categorical features in parallel
plt.figure(figsize=(10, 5))
sns.boxplot(x='Payment_option', y='Price', hue='Vendor_lock-in', data=df)
plt.show()

# Categorical variables to map
categ_list_binary = ['Cluster_management_fee', 'Regional_redundancy', 'Vendor_lock-in', 'Instance_Type',
                     'Hybrid&multi-cloud_support', 'Pay_per_pod_usage', 'Built-in_authentication',
                     'self-recovery_features', 'automate_backup_tasks', 'Monitoring&logging', 'Versioning&upgrades']


# 'Auto-scaling', 'Term_Length', 'Payment_option'

# Defining the map function
def binary_map(x):
    return x.map({'yes': 1, "no": 0, "On Demand": 0, "Spot": 1})


# Applying the function to df
df[categ_list_binary] = df[categ_list_binary].apply(binary_map)
df.head()

# Map Categorical variables with 3 options
categ_list = ['Auto-scaling', 'Term_Length', 'Payment_option']
status = pd.get_dummies(df[categ_list])

status.head()

# Add the above results to the original dataframe df
df = pd.concat([df, status], axis=1)
df.drop(['Auto-scaling', 'Term_Length', 'Payment_option'], axis=1,
        inplace=True)  # drop the initial categorical variables as we have created dummies

df.head()
#
# Rescale the features
# rescale the variables so that they have a comparable scale. If we don't have comparable scales, then some of the
# coefficients as obtained by fitting the regression model might be very large or very small as compared to the other
# coefficients. Use standardization or normalization so that the units of the coefficients obtained are all on the same scale
# We can  use Min-Max scaling or Standardization
#
scaler = MinMaxScaler()

# Apply scaler to all the numeric columns
num_vars = ['Price', 'CPU', 'RAM', 'STORAGE']

df[num_vars] = scaler.fit_transform(df[num_vars])
df.head()

print('Describe the dataframe after standardization')
print(df.describe())

# Check the correlation coefficients to see which variables are highly correlated
plt.figure(figsize=(16, 10))
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
plt.show()

y = df.Price
x_stage = df.drop('Price', axis=1)
x = x_stage.drop('Provider', axis=1)

print(x.info())

# Calculation for p value and other statistical
X = np.column_stack((df['CPU'], df['RAM'], df['STORAGE'], df['Cluster_management_fee'],
                     df['Regional_redundancy'], df['Auto-scaling_both'], df['Auto-scaling_horizontal'],
                     df['Auto-scaling_vertical'], df['Vendor_lock-in'], df['Term_Length_1 Year commitment'],
                     df['Term_Length_3 Year commitment'], df['Term_Length_No commitment'],
                     df['Payment_option_All upfront'], df['Payment_option_no upfront'], df['Payment_option_partially upfront'],
                     df['Instance_Type'], df['Hybrid&multi-cloud_support'], df['Pay_per_pod_usage'],
                     df['Built-in_authentication'], df['self-recovery_features'], df['automate_backup_tasks'],
                     df['Monitoring&logging'], df['Versioning&upgrades']))
Y = df['Price']# scaler = MinMaxScaler()
#
# # Apply scaler to all the numeric columns
# num_vars = ['Price', 'CPU', 'RAM', 'STORAGE']
#
# df[num_vars] = scaler.fit_transform(df[num_vars])
# df.head()
#
# print('Describe the dataframe after standardization')
# print(df.describe())


x2 = sm.add_constant(X)
model_sm = sm.OLS(Y, x2)
results = model_sm.fit()

print(results.summary())
