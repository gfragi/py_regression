# ============== Import libraries =========
import numpy as np
import pandas as pd
import warnings
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype
from scipy import stats
from yellowbrick.regressor import ResidualsPlot
from sklearn import linear_model

import my_functions as mf

warnings.filterwarnings('ignore')  # it is used for some minor warnings in seaborn

vif_features = False  # if only we want to run features with vif < 20
network = False

# %% ============= Load the Data ============================================================

df = mf.load_data('datasets/paas_data.csv')
uniqueList = tuple((column,) for column in df)
for column in df:
    print(df[column].value_counts())

# %% ============= Dataframe Checks before regression ============================================================

mf.dataframe_info(df)  # print some info about df
mf.check_null(df)  # check for null values in the dataframe.

# mf.visualize_data(df)


# %% TODO: for loop to identify th unique values
# columnLength: list = df.shape[1]
# uniqueList: list = []
#
# for x in columnLength:
#     uniqueList = df.iloc[:, x]
if network:
    df.drop(columns=['Provider', 'Deployment Type'], axis=1, inplace=True)
else:
    df.drop(columns=['Provider', 'Deployment Type', 'internal_egress', 'external_egress'], axis=1, inplace=True)

# %% Map binary categorical columns to numerical

categorical_binary = ['Autoscaling', 'Scaling_to_zero', 'OS', 'AppService_Domain', 'Regional_Redudancy',
                      'Container_support']
df[categorical_binary] = df[categorical_binary].apply(mf.binary_map)

# Map>3 categorical columns to numerical

# Write the categorical values as a list
categorical = ['Instance_Type', 'Region', 'Certificates']
categorical2numeric = pd.get_dummies(df[categorical], drop_first=True, sparse=False)

# Add the above results to the original dataframe df
df = pd.concat([df, categorical2numeric], axis=1)
df.drop(columns=categorical, axis=1, inplace=True)

# # %% =============== Log transformation ======================================
# Columns with numerical values to change scale
col2log = []
if network:
    col2log = ['PaaS_Price', 'CPU', 'RAM', 'STORAGE', 'external_egress', 'internal_egress', 'Term_Length']
else:
    col2log = ['CPU', 'RAM', 'STORAGE', "Term_Length"]
               # 'Autoscaling', 'Scaling_to_zero', 'OS', 'AppService_Domain',
               # 'Regional_Redudancy', 'Container_support']

# for column in df:
#     df[column] = df[column].astype(int)


df[col2log] = np.log10(df[col2log] + 1)
df[col2log].replace([col2log], inplace=True)


# %% ===================== Correlation ===========================
# Check the correlation coefficients to see which variables are highly correlated
mf.correlation_triangle(df)

# Check the correlation of the feature with price
mf.corr_per_value(df, 'PaaS_Price')

# %% ===================== Model Evaluation ===========================
y = df.PaaS_Price
x = df.drop('PaaS_Price', axis=1)

mf.model_evaluation(x, y)
model = linear_model.LinearRegression()

# %% =================== Calculate VIF Factors =====================

vif = mf.vif_calc(x)

# %%  =================== Drop columns after vif/reg calculation =====================
if vif_features:
    good_vif = vif[vif['VIF Factor'] < 20]
    drop_after_vif = good_vif['features'].tolist()
    df_new = df[drop_after_vif]
    price = df['PaaS_Price']
    df_new = df_new.join(price)
    df = df_new
    # Re-assign x, y for regression
    y = df.PaaS_Price
    x = df.drop('PaaS_Price', axis=1)
    good_vif.to_csv(f'results/term/paas_vif_net_{network}.csv', index=False)
else:
    vif.to_csv(f'results/term/paas_good_vif_net_{network}.csv', index=False)

# %%============ Detailed calculation for statistical metrics with OLS (Ordinary Least Squares) ==============

# mf.ols_regression(x, y)
x = sm.add_constant(x)
model_sm = sm.OLS(y, x)
results = model_sm.fit()
print(results.summary())
print(results.params)
metrics = pd.read_html(results.summary().tables[0].as_html(), header=0, index_col=0)[0]
coefficients = pd.read_html(results.summary().tables[1].as_html(), header=0, index_col=0)[0]

# ========== Export OLS results =========
if vif_features:
    metrics.to_csv(f'results/term/(vif)paas_reg_metrics_net_{network}.csv', index=True)
    coefficients.to_csv(f'results/term/(vif)paas_reg_coeff_net_{network}.csv', index=True)

else:
    metrics.to_csv(f'results/term/paas_reg_metrics_net_{network}.csv', index=True)
    coefficients.to_csv(f'results/term/paas_reg_coeff_net_{network}.csv', index=True)

# %% ======================== Tornado diagram ======================================
coeff = results.params
coeff = coeff.iloc[(coeff.abs() * -1.0).argsort()]
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot(coeff.values, coeff.index, orient='h', ax=ax, palette="flare", capsize=None)
plt.title(f'Coefficients - PaaS', fontdict=None, loc='center', pad=None)
if vif_features:
    plt.savefig(f'results/term/(vif)paas_coeff_net_{network}.png')
else:
    plt.savefig(f'results/term/(vif)paas_coeff_net_{network}.png')
plt.show()

# ================= Selection of features by P-value ===========================

coeff_results = mf.load_data('results/term/paas_reg_coeff_net_False.csv')
coeff_results.rename(columns={'Unnamed: 0': 'Feature'}, inplace=True)

significant = coeff_results[coeff_results['P>|t|'] < 0.05]

features_list = significant['Feature'].tolist()
features_list.remove('const')
# features_list.remove('AppService_Domain')
features_list.insert(0, 'PaaS_Price')

# features_list.insert(0, 'RAM')

df = df[features_list]

# %%============ Detailed calculation for statistical metrics with OLS (Ordinary Least Squares) ==============

y = df.PaaS_Price
x = df.drop('PaaS_Price', axis=1)

# mf.ols_regression(x, y)
x = sm.add_constant(x)
model_sm = sm.OLS(y, x)
results = model_sm.fit()
print(results.summary())
print(results.params)
metrics_sign = pd.read_html(results.summary().tables[0].as_html(), header=0, index_col=0)[0]
coefficients_sign = pd.read_html(results.summary().tables[1].as_html(), header=0, index_col=0)[0]

# %% ========== Export OLS results =========

metrics.to_csv(f'results/term/significant_paas_reg_metrics.csv', index=True)
coefficients.to_csv(f'results/term/significant_paas_reg_coeff.csv', index=True)

# %% ======================== Tornado diagram ======================================
coeff = results.params
coeff = coeff.iloc[(coeff.abs() * -1.0).argsort()]
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot(coeff.values, coeff.index, orient='h', ax=ax, palette="flare", capsize=None)
plt.title(f'Coefficients - PaaS', fontdict=None, loc='center', pad=None)
# if vif_features:
#     plt.savefig(f'results/term/(vif)paas_coeff_net_{network}.png')
# else:
#     plt.savefig(f'results/term/(vif)paas_coeff_net_{network}.png')
plt.show()

# %% ====== Visualizations for evaluation purposes ===========================
sns.distplot(results.resid, fit=stats.norm, hist=True)
plt.show()

sm.graphics.influence_plot(results, size=40, criterion='cooks', plot_alpha=0.75, ax=None)
plt.show()

visualizer = ResidualsPlot(model, hist=False, qqplot=True)
visualizer.fit(x, y)
# visualizer.score(x_test, y_test)
visualizer.show()
