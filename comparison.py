# ============== Import libraries =========
import numpy as np
import pandas as pd
import warnings
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import math

import seaborn as sns

sns.set()

import my_functions as mf

# %% =========================== Load the Data ============================================================
technology = ['paas', 'iaas', 'caas']
appended_data = []

for i in technology:
    df = mf.load_data(f'results/{i}_significant_coeff.csv')

    # Transpose rows to columns
    df = df.transpose()
    df.columns = df.iloc[0]
    df.drop(labels='Unnamed: 0', axis=0, inplace=True)
    df['Technology'] = i  # Add the year column
    # Append the data after each iteration
    appended_data.append(df)

# %% =========================== Create the Dataframes ============================================================

# Create one Dataframe with results
df = pd.concat(appended_data)

# Dataframe for useful metrics
coeffPerTechnology = df.loc['coef']

# Drop columns with  at least nan values
coeffPerTechnology_na = coeffPerTechnology.dropna(axis=1)

# ==================== Visualization ============================================================================
# Create plot to compare the results of coefficients
# cols = list(coeffPerTechnology_na.columns.values)
# cols.remove('Technology')
# colors = ('red', 'yellow', 'blue', 'green', 'orange', 'olive', 'magenta', 'cyan', 'grey')
features = {'CPU': 'red', 'RAM': 'blue', 'STORAGE': 'green', 'Term_Length': 'orange',
            'Instance_Type_dedicated': 'olive', 'Region_US': 'magenta', 'OS_Windows': 'cyan'}

fig, ax = plt.subplots(figsize=(15, 10))
x_value = 'Technology'

for key in features.keys():
    coeffPerTechnology_na.plot(x=x_value, y=key, color=features[key], kind='line', ax=ax, marker="o")

plt.xlabel('Technology')
plt.ylabel('Coefficient')
plt.legend(ncol=2, loc='upper left')
# displaying the title
plt.title(f'coefficients per technology')
# plt.savefig(f'results/amazon/{subfolder}_coef_evolution')
plt.show()
