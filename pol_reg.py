import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


### Provide data to work with

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])

### Transform input data. Transform the array of inputs to include non-linear terms such as 𝑥²

transformer = PolynomialFeatures(degree=2, include_bias=False).fit(x)

x_ = transformer.transform(x)

print(x_)

### Create a model and fit it

model = LinearRegression().fit(x_, y)

### Get the results

r_sq = model.score(x_, y)

print('coefficient of determination:', r_sq)

#The attributes of model are .intercept_, which represents the coefficient, 𝑏₀ and .coef_, which represents 𝑏₁
print('intercept:', model.intercept_)
print('slope:', model.coef_)


###  Predict response

y_pred = model.predict(x_)
print('predicted response:', y_pred, sep='\n')