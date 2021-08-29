# ============== Import libraries =========
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


warnings.filterwarnings('ignore')  # it is used for some minor warnings in seaborn

# ============= Load the Data ============================================================
#%% Load the csv & print columns' info
# df = pd.read_csv('cn_provider_pricing_dummy.csv')  # dummy data
df = pd.read_csv('cn_pricing_per_provider.csv')  # real data

# Drop some not useful for calculation columns (sum calculation for total price)
df = df.drop(['CPU_RAM_Price', 'Storage_Price', 'Cluster_fee', 'licensed_OS', 'Hybrid_support'], axis=1)

# Convert the price unit to $/month from $/hour
df['Price'] = df['Price']
# print(df['Price'])

# print('rows x columns:', df.shape)
# print('Columns info:', df.info())
# print('Data highlights: \n', df.describe())

# Check for null values
# print(df.isnull().sum() * 100 / df.shape[0])

# =========== Visualize the Data ======================================
#%% Visualize numeric variables
ax = sns.pairplot(df)
ax.fig.suptitle('Visualize numeric variables')
plt.plot(color='green')
# plt.show()

# Visualize categorical variables
fig = plt.figure(figsize=(20, 28))
fig.suptitle('Outlier analysis for categorical variables', fontsize=32)

plt.subplot(5, 3, 1)
sns.boxplot(x='Cluster_management_fee', y='Price', data=df)
sns.swarmplot(x='Cluster_management_fee', y='Price', data=df, color=".25")

plt.subplot(5, 3, 2)
sns.boxplot(x='Regional_redundancy', y='Price', data=df)
sns.swarmplot(x='Regional_redundancy', y='Price', data=df, color=".25")

plt.subplot(5, 3, 3)
sns.boxplot(x='Autoscaling', y='Price', data=df)
sns.swarmplot(x='Autoscaling', y='Price', data=df, color=".25")

plt.subplot(5, 3, 4)
sns.boxplot(x='Vendor_lock-in', y='Price', data=df)
sns.swarmplot(x='Vendor_lock-in', y='Price', data=df, color=".25")

plt.subplot(5, 3, 5)
sns.boxplot(x='Payment_option', y='Price', data=df)
sns.swarmplot(x='Payment_option', y='Price', data=df, color=".25")

plt.subplot(5, 3, 6)
sns.boxplot(x='Term_Length', y='Price', data=df)
sns.swarmplot(x='Term_Length', y='Price', data=df, color=".25")

plt.subplot(5, 3, 7)
sns.boxplot(x='Instance_Type', y='Price', data=df)
sns.swarmplot(x='Instance_Type', y='Price', data=df, color=".25")

plt.subplot(5, 3, 8)
sns.boxplot(x='Disk_type', y='Price', data=df)
sns.swarmplot(x='Disk_type', y='Price', data=df, color=".25")

plt.subplot(5, 3, 9)
sns.boxplot(x='OS', y='Price', data=df)
sns.swarmplot(x='OS', y='Price', data=df, color=".25")

plt.subplot(5, 3, 10)
sns.boxplot(x='Hybrid_multicloud_support', y='Price', data=df)
sns.swarmplot(x='Hybrid_multicloud_support', y='Price', data=df, color=".25")

plt.subplot(5, 3, 11)
sns.boxplot(x='Pay_per_pod_usage', y='Price', data=df)
sns.swarmplot(x='Pay_per_pod_usage', y='Price', data=df, color=".25")

plt.subplot(5, 3, 12)
sns.boxplot(x='Region', y='Price', data=df)
sns.swarmplot(x='Region', y='Price', data=df, color=".25")
# plt.show()

# Visualize categorical features in parallel, we could add more
#%%
plt.figure(figsize=(10, 5))
sns.boxplot(x='Hybrid_multicloud_support', y='Price', hue='OS', data=df, width=0.5)
# plt.show()

# =========== Data preparation =================

# Categorical variables to map
category_list_binary = ['Cluster_management_fee', 'Vendor_lock-in', 'Disk_type', 'Hybrid_multicloud_support',
                        'Pay_per_pod_usage', 'Built-in_authentication', 'self-recovery_features',
                        'automate_backup_tasks', 'Versioning&upgrades']


# Defining the map function
def binary_map(k):
    return k.map({'yes': 1, 'no': 0, 'Standard': 0, 'SSD': 1})


# Applying the function to df
df[category_list_binary] = df[category_list_binary].apply(binary_map)
df.head()

# Map Categorical variables with 3 options
category_list = ['Autoscaling', 'Term_Length', 'Payment_option', 'OS', 'Instance_Type', 'Region']
status = pd.get_dummies(df[category_list])

status.head()

# Add the above results to the original dataframe df
df = pd.concat([df, status], axis=1)
df.drop(['Autoscaling', 'Term_Length', 'Payment_option', 'OS', 'Instance_Type', 'Region'], axis=1,
        inplace=True)  # drop the initial categorical variables as we have created dummies

# Drop options
#%% Drop the columns-features.

df.drop(['Built-in_authentication', 'self-recovery_features', 'automate_backup_tasks', 'Versioning&upgrades', 'STORAGE',
         'Regional_redundancy', 'Payment_option_no upfront', 'OS_Windows', 'Autoscaling_horizontal', 'Autoscaling_both',
         'Region_Australia', 'Region_Africa', 'Payment_option_All upfront', 'Vendor_lock-in', 'Payment_option_partially upfront',
         'Pay_per_pod_usage', 'Term_Length_1 Year commitment', 'Disk_type', 'OS_Linux', 'Region_Asia', 'Region_US',
         'Region_South America', 'Instance_Type_On Demand', 'Term_Length_3 Year commitment', 'Instance_Type_Spot', 'OS_free'], axis=1, inplace=True)

# ===================== Correlation ===========================

#%% Check the correlation coefficients to see which variables are highly correlated
correlation_method: str = 'pearson'

corr = df.corr(method=correlation_method)
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
f, ax = plt.subplots(figsize=(32, 16))
heatmap = sns.heatmap(corr, mask=mask, annot=True, cmap=cmap, fmt=".2f")
heatmap.set_title(f"Triangle Correlation Heatmap - {correlation_method}", fontdict={'fontsize': 18}, pad=16)
plt.savefig('plots/heatmap_triangle.png')
# plt.show()

y = df.Price
x_stage = df.drop('Price', axis=1)
x = x_stage.drop('Provider', axis=1)

# print(x.info())

#%% Features Correlating with Price

plt.figure(figsize=(12, 15))
heatmap = sns.heatmap(df.corr(method=correlation_method)[['Price']].sort_values(by='Price', ascending=False), vmin=-1, vmax=1, annot=True,
                      cmap='BrBG')
heatmap.set_title(f"Features Correlating with Price - {correlation_method}", fontdict={'fontsize': 18}, pad=16)
plt.savefig('plots/heatmap_only_price.png')
# plt.show()


####### Positive Correlation ######## https://towardsdatascience.com/simple-and-multiple-linear-regression-with-python-c9ab422ec29c
# 1–0.8 → Very strong
# 0.799–0.6 → Strong
# 0.599–0.4 → Moderate
# 0.399–0.2 → Weak
# 0.199–0 → Very Weak

# #%% regression plot using seaborn - Very strong
# fig = plt.figure(figsize=(10, 7))
# sns.regplot(x=df.CPU, y=df.Price, color='#619CFF', marker='o')
#
# # legend, title, and labels.
# plt.legend(labels=['CPU'])
# plt.title('Relationship between Price and CPU', size=20)
# plt.xlabel('CPU(Cores)', size=18)
# plt.ylabel('Price ($/hour)', size=18)
# plt.show()
#
# fig = plt.figure(figsize=(10, 7))
# sns.regplot(x=df.STORAGE, y=df.Price, color='#619CFF', marker='o')
#
# # legend, title, and labels.
# plt.legend(labels=['Storage'])
# plt.title('Relationship between Price and Storage', size=20)
# plt.xlabel('Storage(GB)', size=18)
# plt.ylabel('Price ($/hour)', size=18)
# plt.show()
#
# #%% regression plot using seaborn - Strong
# fig = plt.figure(figsize=(10, 7))
# sns.regplot(x=df.RAM, y=df.Price, color='#619CFF', marker='o')
#
# # legend, title, and labels.
# plt.legend(labels=['RAM'])
# plt.title('Relationship between Price and RAM', size=20)
# plt.xlabel('RAM(GB)', size=18)
# plt.ylabel('Price ($/hour)', size=18)
# plt.show()
#
# #%% regression plot using seaborn - Weak
# fig = plt.figure(figsize=(10, 7))
# sns.regplot(x=df.Hybrid_multicloud_support, y=df.Price, color='#619CFF', marker='o')
#
# # legend, title, and labels.
# plt.legend(labels=['Hybrid_multicloud_support'])
# plt.title('Relationship between Price and Hybrid_multicloud_support', size=20)
# plt.xlabel('Hybrid_multicloud_support', size=18)
# plt.ylabel('Price ($/hour)', size=18)
# plt.show()
#
#
# fig = plt.figure(figsize=(10, 7))
# sns.regplot(x=df.Hybrid_multicloud_support, y=df.OS_Linux, color='#619CFF', marker='o')
#
# # legend, title, and labels.
# plt.legend(labels=['Hybrid_multicloud_support'])
# plt.title('Relationship between OS_Linux and Hybrid_multicloud_support', size=20)
# plt.xlabel('Hybrid_multicloud_support', size=18)
# plt.ylabel('OS_linux', size=18)
# plt.show()
#
# #%% regression plot using seaborn - Very Weak
# fig = plt.figure(figsize=(10, 7))
# sns.regplot(x=df.Regional_redundancy, y=df.Price, color='#619CFF', marker='o')
#
# # legend, title, and labels.
# plt.legend(labels=['Regional_redundancy'])
# plt.title('Relationship between Price and Regional_redundancy', size=20)
# plt.xlabel('Regional_redundancy', size=18)
# plt.ylabel('Price ($/hour)', size=18)
# plt.show()
#
# #%% regression plot using seaborn - Negative
# fig = plt.figure(figsize=(10, 7))
# sns.regplot(x=df.Cluster_management_fee, y=df.Price, color='#619CFF', marker='o')
#
# # legend, title, and labels.
# plt.legend(labels=['Regional_redundancy'])
# plt.title('Relationship between Price and Cluster_management_fee', size=20)
# plt.xlabel('Cluster_management_fee', size=18)
# plt.ylabel('Price ($/hour)', size=18)
# plt.show()

# ================ Model Evaluation ===========================
#%% Evaluate the model performance, split the the dataset into 2 partitions (80% - 20% ration)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Apply linear regression to train set
model = linear_model.LinearRegression()
model.fit(x_train, y_train)

# Apply trained model to train dataset
y_pred_train = model.predict(x_train)

print('\n======== TRAIN dataset - 80% ===========')
print('Coefficients:\n', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.3f'
      % mean_squared_error(y_train, y_pred_train))
print('Coefficient of determination (R^2): %.3f\n'
      % r2_score(y_train, y_pred_train))

# Apply trained model to test dataset
y_pred_test = model.predict(x_test)

print('\n========= TEST dataset - 20% ===========')
print('Coefficients:\n', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.3f'
      % mean_squared_error(y_test, y_pred_test))
print('Coefficient of determination (R^2): %.3f\n'
      % r2_score(y_test, y_pred_test))

# Evaluation Plots
plt.figure(figsize=(11, 5))

# 1 row, 2 column, plot 1
plt.subplot(1, 2, 1)
plt.scatter(x=y_train, y=y_pred_train, c="#7CAE00", alpha=0.3)

# Add trendline
z = np.polyfit(y_train, y_pred_train, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), "#F8766D")

plt.ylabel('Predicted prices')
plt.xlabel('Actual prices')

# 1 row, 2 column, plot 2
plt.subplot(1, 2, 2)
plt.scatter(x=y_test, y=y_pred_test, c="#619CFF", alpha=0.3)

z = np.polyfit(y_test, y_pred_test, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), "#F8766D")

plt.ylabel('Predicted prices')
plt.xlabel('Actual prices')

# plt.savefig('plots/plot_horizontal_logS.png')
# plt.savefig('plots/plot_horizontal_logS.pdf')
plt.show()

# ============ Detailed calculation for statistical metrics with OLS (Ordinary Least Squares) ==============

x = sm.add_constant(x)
model_sm = sm.OLS(y, x)
results = model_sm.fit()

print(results.summary())
