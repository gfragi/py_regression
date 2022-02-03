# ============== Import libraries =========
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def load_data(self: object) -> object:
    dataframe = pd.read_csv(self)
    return dataframe


def select_year(year, dataframe):
    dataframe = dataframe.loc[dataframe['year'] == year]
    return dataframe


def dataframe_info(df):
    print('rows x columns: \n', df.shape,
          'Columns info: \n', df.info(),
          'Data highlights: \n', df.describe())


def check_null(df):
    print('Null values per column: \n', df.isnull().sum() * 100 / df.shape[0])


def binary_map(k):
    return k.map({'yes': 1, 'no': 0, 'Dedicated': 0, 'Shared': 1, 'No Upfront': 0,
                  'Partial Upfront': 1, 'standard': 0, 'convertible': 1, 'Yes': 1, 'No': 0,
                  'Windows': 1, 'Linux': 0})


def log_numerical(dataframe=None, *col):
    dataframe[col] = np.log10(dataframe[col] + 1)
    return dataframe


def correlation_triangle(dataframe, year=None, folder=None, subfolder=None, corr_method='pearson'):
    corr = dataframe.corr(method=corr_method)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    ax = plt.subplots(figsize=(30, 18))
    heatmap = sns.heatmap(corr, mask=mask, annot=True, cmap=cmap, fmt=".2f")
    heatmap.set_title(f"Triangle Correlation Heatmap - {year}", fontdict={'fontsize': 24}, pad=1)
    # plt.savefig(f'results/{folder}/{subfolder}/{year}_triangle.png')
    plt.show()
    return ax


def corr_per_value(dataframe, value, year=None, corr_method='pearson'):
    plt.figure(figsize=(12, 15))
    heatmap = sns.heatmap(dataframe.corr(method=corr_method)[[value]].sort_values(by=value, ascending=False),
                          vmin=-1,
                          vmax=1, annot=True,
                          cmap='BrBG')
    heatmap.set_title(f"Features Correlating with Price - {year}", fontdict={'fontsize': 18}, pad=16)
    plt.show()
    return heatmap


def model_evaluation(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    print('\n======== TRAIN dataset - 80% ===========')
    print('Coefficients:\n', model.coef_)
    print('Intercept:', model.intercept_)
    print('Mean squared error (MSE): %.3f'
          % mean_squared_error(y_train, y_pred_train))
    print('Coefficient of determination (R^2): %.3f\n'
          % r2_score(y_train, y_pred_train))
    print('\n========= TEST dataset - 20% ===========')
    print('Coefficients:\n', model.coef_)
    print('Intercept:', model.intercept_)
    print('Mean squared error (MSE): %.3f'
          % mean_squared_error(y_test, y_pred_test))
    print('Coefficient of determination (R^2): %.3f\n'
          % r2_score(y_test, y_pred_test))
    return model


def ols_regression(x, y):
    x = sm.add_constant(x)
    model_sm = sm.GLS(y, x)
    results = model_sm.fit()
    print(results.summary())
    print(results.params)
    return results


def gls_regression(x, y):
    x = sm.add_constant(x)
    model_sm = sm.GLS(y, x)
    results = model_sm.fit()
    print(results.summary())
    print(results.params)
    return results


def vif_calc(x):
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    vif["features"] = x.columns
    return vif.round(1)


def visualize_data(dataframe):
    num_list = []
    cat_list = []
    for column in dataframe:
        ax = plt.figure(column, figsize=(8, 5))
        plt.title(column)
        if is_numeric_dtype(dataframe[column]):
            dataframe[column].plot(kind='hist')
            num_list.append(column)
        elif is_string_dtype(dataframe[column]):
            dataframe[column].value_counts().plot(kind='barh', color='#43FF76')
            cat_list.append(column)
        plt.xlabel('Bundles')
    plt.show()
    return ax, plt.show()


def unique_values():
    pass


def reorder_columns(dataframe, col_name, position):
    """Reorder a dataframe's column.
    Args:
        dataframe (pd.DataFrame): dataframe to use
        col_name (string): column name to move
        position (0-indexed position): where to relocate column to
    Returns:
        pd.DataFrame: re-assigned dataframe
    """
    temp_col = dataframe[col_name]
    dataframe = dataframe.drop(columns=[col_name])
    dataframe.insert(loc=position, column=col_name, value=temp_col)
    return dataframe
