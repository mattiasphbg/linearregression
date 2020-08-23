# If the goal is prediction, forecasting, or error reduction,
# [clarification needed] linear regression can be used to fit a predictive model
# to an observed data set of values of the response and explanatory variables.

# you need regression to answer whether and how some phenomenon influences the other or
# how several variables are related.

import numpy as np  # Python library that provides a multidimensional array object
from sklearn.linear_model import LinearRegression  # Simple and efficient tools for predictive data analysis

model = LinearRegression()
# creates the variable model as the instance of LinearRegression. The line in between.

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
# You should notice that you can provide y as a two-dimensional array as well.
# In this case, you’ll get a similar result.

# .intercept_ is a one-dimensional array with the single element 𝑏₀
# .coef_ is a two-dimensional array with the single element 𝑏₁ and vise versa.
# ---------------------------------------------------------------
# this array is required to be two-dimensional,
# or to be more precise, to have one column and as many rows as necessary.

y = np.array([5, 20, 14, 32, 22, 38])
# makes 1-d on the y-axis


model.fit(x, y)
# With .fit(), you calculate the optimal values of the weights 𝑏₀ and 𝑏₁,
# using the existing input and output (x and y) as the arguments. In other words,.fit() fits the model.


# model = LinearRegression().fit(x, y) Does the same thing as  model.fit() and Model = linear

r_sq = model.score(x, y)  # You can obtain the coefficient of determination (𝑅²) with .score()

# ----------------------------------------------------------------

# print('intercept:', model.intercept_)
# intercept: 5.633333333333329
# print('slope:', model.coef_)
# slope: [0.54]


# Intercept may refer to:
# X-intercept, the point where a line crosses the x-axis
# Y-intercept, the point where a line crosses the y-axis

# slope - defined as the change in the y coordinate divided by the corresponding change in the x coordinate.
# The rate at which an ordinate (y-axis) of a point of a line on a coordinate plane changes with respect to a change in
# the abscissa(x-axis)


# The code above illustrates how to get 𝑏₀ and 𝑏₁.
# You can notice that .intercept_ is a scalar, while .coef_ is an array.

# The value 𝑏₀ = 5.63 (approximately) illustrates that your model predicts the response 5.63 when 𝑥 is zero.
# The value 𝑏₁ = 0.54 means that the predicted response rises by 0.54 when 𝑥 is increased by one.


# b1 - This is the SLOPE of the regression line. Thus this is the amount that the Y variable (dependent) will change
# for each 1 unit change in the X variable.
# b0 - This is the intercept of the regression line with the y-axis. In other words it is the value of Y if the value of
# X = 0.
# Y-hat = b0 + b1(x) - This is the sample regression line. You must calculate b0 & b1 to create this line.
# Y-hat stands for the predicted value of Y, and it can be obtained by plugging an individual value of x into the
# equation and calculating y-hat.

# --------------------------------------------------------------------
#  y_pred = model.predict(x) is the same thing as that below.

y_pred = model.intercept_ + model.coef_ * x
print('predicted response: ', y_pred)
