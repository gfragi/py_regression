#Following the https://realpython.com/linear-regression-in-python/

# Import the packages and classes
from os import sep
import numpy as np
from sklearn.linear_model import LinearRegression


## y = 𝑏₀ + 𝑏₁x

### Provide data to work with and eventually do appropriate transformations


# call .reshape() on x because this array is required to be two-dimensional = one column and as many rows as necessary
x=np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1)) 

y=np.array([5, 20, 14, 32, 22, 38])


### Create a regression model and fit it with existing data

model=LinearRegression().fit(x, y)          # with .fit calculate the optimal values of the weights 𝑏₀ and 𝑏₁


### Check the results of model fitting to know whether the model is satisfactory.

r_sq = model.score(x, y)        # obtain the coefficient of determination (𝑅²) with .score()

print('coefficient of determination:', r_sq)

#The attributes of model are .intercept_, which represents the coefficient, 𝑏₀ and .coef_, which represents 𝑏₁
print('intercept:', model.intercept_)
print('slope:', model.coef_)

### Apply the model for predictions.

y_pred = model.predict(x)     ## same think to     "y_pred = model.intercept_ + model.coef_ * x"

print('predicted response:', y_pred)