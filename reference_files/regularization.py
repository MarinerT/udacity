# TODO: Add import statements
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso


# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = np.genfromtxt("data.csv",  delimiter=',')
X = train_data[:,:6]
y = train_data[:,-1]

# TODO: Create the linear regression model with lasso regularization.
lasso_reg = Lasso()

# TODO: Fit the model.
lasso_reg.fit(X,y)

# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)
