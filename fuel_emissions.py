from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler, MaxAbsScaler, MinMaxScaler

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn import preprocessing, metrics, model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =============================================================================
#                        PREPROCESSING
# =============================================================================     

# Load the dataset
X = pd.read_csv('fuel_emissions.csv')
print(X.shape)


# # find total NaN values for each column of data
# for column in X:
#     print("total NaN values in column: ", column, "   ", X[column].isnull().sum())
    
    
# drop columns that contains greater than 100 NaN values + file + description.
drop_columns = ["file", "description", "tax_band", "thc_emissions", "transmission_type", 
                "thc_nox_emissions", 
                "particulates_emissions", "standard_12_months",
                "standard_6_months", "first_year_12_months", "first_year_6_months"]

X = X.drop(columns = drop_columns)
print(X.shape)

# after deleting columns,  delete also the rows which contains NaN values 
# these rows are:
X.dropna(axis = 0, how = 'any', inplace = True)
print(X.shape)


# ------------ drop target "fuel_cost_12000_miles"---------------------
y = X["fuel_cost_12000_miles"]
X = X.drop(columns = ["fuel_cost_12000_miles"])


# split train and test set
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.25, random_state = 0)

# One hot encoding for categorical columns  'model', 'manufacturer', 'fuel_type', 'transmission'
categorical_columns = ['model', 'manufacturer', 'fuel_type', 'transmission']

enc = OneHotEncoder(categories = 'auto', handle_unknown = 'ignore')
encoded = pd.DataFrame(enc.fit_transform(X_train[categorical_columns]).toarray())
X_train = X_train.join(encoded).drop(columns = categorical_columns)

encoded = pd.DataFrame(enc.transform(X_test[categorical_columns]).toarray())
X_test = X_test.join(encoded).drop(columns = categorical_columns)


imputer = SimpleImputer(strategy = "mean")
imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)


scaler = MinMaxScaler(feature_range = (-1, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




# =============================================================================
#                        REGRESSION
# =============================================================================     

#reg = LinearRegression(copy_X = True, fit_intercept = True, n_jobs = -1)
#reg = DecisionTreeRegressor(criterion = "friedman_mse", random_state = 0)
reg = LinearSVR(C = 1, random_state = 0)
#reg = KNeighborsRegressor(n_neighbors = 2)
#reg = RandomForestRegressor(n_estimators = 400, criterion = "mse", n_jobs = -1)


# fit and predict
reg.fit(X_train, y_train)
y_predicted = reg.predict(X_test)


#print('Coefficients: ', linearRegressionModel.coef_)
# The mean squared error
print('Mean squared error: %.2f' %mean_squared_error(y_test, y_predicted))
print('Coefficient of determination: %.2f' % r2_score(y_test, y_predicted))
print("RMSE: ", mean_squared_error(y_test, y_predicted, squared = False))
print("MAE: ", mean_absolute_error(y_test, y_predicted))



      










