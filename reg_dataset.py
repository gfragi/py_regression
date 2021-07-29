import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# Assign the url where the dataset reside 
delaney_url = 'https://raw.githubusercontent.com/dataprofessor/data/master/delaney.csv'

# Read the dataser and assign it to a variable
delaney_df = pd.read_csv(delaney_url)

# Assign the url where the dataset reside 
delaney_descriptors_url = 'https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv'

# Read the calculated descriptors
delaney_descriptors_df = pd.read_csv(delaney_descriptors_url)

print(delaney_descriptors_df)

# Drop logS which is the y variable
x = delaney_descriptors_df.drop('logS', axis=1)

# Assign to y variable the logS column
y = delaney_descriptors_df.logS

# In evaluate the model performance split e the dataset into 2 partitions (80% - 20% ration)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Apply linear regression to train set
model = linear_model.LinearRegression()
model.fit(x_train, y_train)


# Apply trained model to train dataset 
y_pred_train = model.predict(x_train)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(y_train, y_pred_train))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(y_train, y_pred_train))


# Apply trained model to test dataset 
y_pred_test = model.predict(x_test)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(y_test, y_pred_test))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(y_test, y_pred_test))


# Print the Regression equation
yintercept = '%.2f' % model.intercept_
LogP = '%.2f LogP' % model.coef_[0]
MW = '%.4f MW' % model.coef_[1]
RB = '%.4f RB' % model.coef_[2]
AP = '%.2f AP' % model.coef_[3]

print('LogS = ' + ' ' + yintercept + ' ' + LogP + ' ' + MW + ' + ' + RB + ' ' + AP)


plt.figure(figsize=(11, 5))

# 1 row, 2 column, plot 1
plt.subplot(1, 2, 1)
plt.scatter(x=y_train, y=y_pred_train, c="#7CAE00", alpha=0.3)

# Add trendline
z = np.polyfit(y_train, y_pred_train, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test),"#F8766D")

plt.ylabel('Predicted LogS')


# 1 row, 2 column, plot 2
plt.subplot(1, 2, 2)
plt.scatter(x=y_test, y=y_pred_test, c="#619CFF", alpha=0.3)

z = np.polyfit(y_test, y_pred_test, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test),"#F8766D")

plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')

plt.savefig('plots/plot_horizontal_logS.png')
plt.savefig('plots/plot_horizontal_logS.pdf')
plt.show()



X = np.column_stack((x['MolLogP'], x['MolWt'], x['NumRotatableBonds'], x['AromaticProportion']))
Y = y['logS']
x2 = sm.add_constant(X)
est = sm.OLS(Y, x2)
est2 = est.fit
print(est2.summary())