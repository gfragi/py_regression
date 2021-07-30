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
print(df.isnull().sum()*100/df.shape[0])


# Outlier Analysis
fig, axs = plt.subplots(2,3, figsize = (10, 3))
plt1 = sns.boxplot(df['Price'], ax = axs[0, 0])
plt2 = sns.boxplot(df['CPU'], ax = axs[0, 1])
plt3 = sns.boxplot(df['RAM'], ax = axs[0, 2])
plt1 = sns.boxplot(df['STORAGE'], ax = axs[1, 0])
# plt2 = sns.boxplot(df['Cluster_management_fee'], ax = axs[1, 1])
# plt3 = sns.boxplot(df['Regional redundancy'], ax = axs[1, 2])
# plt1 = sns.boxplot(df['Auto-scaling'], ax = axs[2, 0])
# plt2 = sns.boxplot(df['Vendor_lock-in'], ax = axs[2, 1])
# plt3 = sns.boxplot(df['Payment_option'], ax = axs[2, 2])
# plt1 = sns.boxplot(df['Term_Length'], ax = axs[3, 0])
# plt2 = sns.boxplot(df['Instance_Type'], ax = axs[3, 1])
# plt3 = sns.boxplot(df['Hybrid&multi-cloud_support'], ax = axs[3, 2])
# plt1 = sns.boxplot(df['Pay_per_pod_usage'], ax = axs[4, 0])
# plt2 = sns.boxplot(df['Built-in_authentication'], ax = axs[4, 1])
# plt3 = sns.boxplot(df['self-recovery_features'], ax = axs[4, 2])
# plt1 = sns.boxplot(df['automate_backup_tasks'], ax = axs[5, 0])
# plt2 = sns.boxplot(df['Monitoring&logging'], ax = axs[5, 1])
# plt3 = sns.boxplot(df['Versioning&upgrades'], ax = axs[5, 2])

plot1 = plt.tight_layout

plt.show(plot1)