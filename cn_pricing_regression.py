import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.formula.api import ols

warnings.filterwarnings('ignore')  # it is used for some minor warnings in seaborn

# Load the csv & print columns' info
df = pd.read_csv('cn_pricing_per_provider.csv')  # real data

# Drop some not useful for calculation columns (sum calculation for total price)
df = df.drop(['CPU_RAM_Price', 'Storage_Price', 'Cluster_fee', 'licensed_OS', 'Hybrid_support'], axis=1)

# Convert the price unit to $/month from $/hour
df['Price'] = df['Price'] * 730
print(df['Price'])

print('rows x columns:', df.shape)
print('Columns info:', df.info())
print('Data highlights: \n', df.describe())

# Check for null values
print(df.isnull().sum() * 100 / df.shape[0])

# # Outlier Analysis for numeric variables
# fig, axs = plt.subplots(1, 4, figsize=(15, 6))
# fig.suptitle('Outlier analysis for numeric variables', fontsize=18)
# plt1 = sns.boxplot(df['Price'], ax=axs[0])
# plt2 = sns.boxplot(df['CPU'], ax=axs[1])
# plt3 = sns.boxplot(df['RAM'], ax=axs[2])
# plt4 = sns.boxplot(df['STORAGE'], ax=axs[3])
#
# plt.tight_layout()
# plt.show()

# Visualize numeric variables
ax = sns.pairplot(df)
ax.fig.suptitle('Visualize numeric variables')
plt.plot(color='green')
plt.show()

# Visualize categorical variables
fig = plt.figure(figsize=(20, 28))
fig.suptitle('Outlier analysis for categorical variables', fontsize=32)
plt.subplot(6, 3, 1)
sns.boxplot(x='Cluster_management_fee', y='Price', data=df)
plt.subplot(6, 3, 2)
sns.boxplot(x='Regional_redundancy', y='Price', data=df)
plt.subplot(6, 3, 3)
sns.boxplot(x='Autoscaling', y='Price', data=df)
plt.subplot(6, 3, 4)
sns.boxplot(x='Vendor_lock-in', y='Price', data=df)
plt.subplot(6, 3, 5)
sns.boxplot(x='Payment_option', y='Price', data=df)
plt.subplot(6, 3, 6)
sns.boxplot(x='Term_Length', y='Price', data=df)
plt.subplot(6, 3, 7)
sns.boxplot(x='Instance_Type', y='Price', data=df)
plt.subplot(6, 3, 8)
sns.boxplot(x='Disk_type', y='Price', data=df)
plt.subplot(6, 3, 9)
sns.boxplot(x='OS', y='Price', data=df)
plt.subplot(6, 3, 10)
sns.boxplot(x='Hybrid_multicloud_support', y='Price', data=df)
plt.subplot(6, 3, 11)
sns.boxplot(x='Pay_per_pod_usage', y='Price', data=df)
plt.subplot(6, 3, 12)
sns.boxplot(x='Region', y='Price', data=df)
plt.subplot(6, 3, 13)
sns.boxplot(x='Built-in_authentication', y='Price', data=df)
plt.subplot(6, 3, 14)
sns.boxplot(x='self-recovery_features', y='Price', data=df)
plt.subplot(6, 3, 15)
sns.boxplot(x='automate_backup_tasks', y='Price', data=df)
plt.subplot(6, 3, 16)
sns.boxplot(x='Versioning&upgrades', y='Price', data=df)
plt.show()

# Visualize categorical features in parallel, we could add more
plt.figure(figsize=(10, 5))
sns.boxplot(x='Payment_option', y='Price', hue='Vendor_lock-in', data=df)
plt.show()

# Categorical variables to map
category_list_binary = ['Cluster_management_fee', 'Regional_redundancy', 'Vendor_lock-in', 'Disk_type',
                        'Hybrid_multicloud_support', 'Pay_per_pod_usage', 'Built-in_authentication',
                        'self-recovery_features', 'automate_backup_tasks', 'Versioning&upgrades']


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

df.head()
#
# Rescale the features
# rescale the variables so that they have a comparable scale. If we don't have comparable scales, then some of the
# coefficients as obtained by fitting the regression model might be very large or very small as compared to the other
# coefficients. Use standardization or normalization so that the units of the coefficients obtained are all on the same scale
# We can  use Min-Max scaling or Standardization
# scaler = StandardScaler()

scaler = MinMaxScaler()

# Apply scaler to all the numeric columns
num_vars = ['Price', 'CPU', 'RAM', 'STORAGE']

df[num_vars] = scaler.fit_transform(df[num_vars])
df.head()

print('Describe the dataframe after rescaling \n')
print(df.describe())

# Check the correlation coefficients to see which variables are highly correlated
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
f, ax = plt.subplots(figsize=(32, 16))
sns.heatmap(corr, mask=mask, annot=True, cmap=cmap, fmt=".2f")

plt.show()

y = df.Price
x_stage = df.drop('Price', axis=1)
x = x_stage.drop('Provider', axis=1)

print(x.info())

# Evaluate the model performance we split the dataset into 2 partitions (80% - 20% ration)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Apply linear regression to train set
model = linear_model.LinearRegression()
model.fit(x_train, y_train)

# Apply trained model to train dataset
y_pred_train = model.predict(x_train)

print('======== TRAIN dataset - 80% ===========')
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(y_train, y_pred_train))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(y_train, y_pred_train))

# Apply trained model to test dataset
y_pred_test = model.predict(x_test)

print('========= TEST dataset - 20% ===========')
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(y_test, y_pred_test))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(y_test, y_pred_test))

# Plots
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

# Calculation for p value and other
X = np.column_stack((df['CPU'], df['RAM'], df['STORAGE'], df['Cluster_management_fee'],
                     df['Regional_redundancy'], df['Vendor_lock-in'], df['Disk_type'], df['Hybrid_multicloud_support'],
                     df['Pay_per_pod_usage'], df['Built-in_authentication'], df['self-recovery_features'],
                     df['automate_backup_tasks'],
                     df['Versioning&upgrades'], df['Autoscaling_both'], df['Autoscaling_horizontal'],
                     df['Term_Length_1 Year commitment'], df['Term_Length_2 Year commitment'],
                     df['Term_Length_3 Year commitment'], df['Term_Length_No commitment'],
                     df['Payment_option_All upfront'],
                     df['Payment_option_no upfront'], df['OS_Linux'], df['OS_Windows'], df['OS_free'],
                     df['Payment_option_partially upfront'], df['Instance_Type_Dedicated'],
                     df['Instance_Type_On Demand'],
                     df['Instance_Type_Spot'], df['Region_Asia'], df['Region_Europe'], df['Region_US'],
                     df['Region_Australia'], df['Region_Africa'],
                     df['Region_South America']))

Y = df['Price']  # scaler = MinMaxScaler()
x2 = sm.add_constant(X)
model_sm = sm.OLS(Y, x2)
results = model_sm.fit()

print(results.summary())



