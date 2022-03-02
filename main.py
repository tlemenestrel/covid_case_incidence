# Libraries
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

# Code
from eda import pearson_corr_mat, get_corr_features
from models import LinearRegression, RidgeRegression, LassoRegression, log_loss
from utils import separate_xy, min_max_normalization, mean_normalization, vif_feature_selection

################################################################################
# DATA PRE-PROCESSING
################################################################################

# Read training data, split in X and y and drop date column
df_train = pd.read_csv('train_data.csv')
df_train = df_train.drop('date', axis=1)

################################################################################
# FEATURE SELECTION
################################################################################

print()
print('#######################################################################')
print('FEATURE SELECTION')
print('#######################################################################')
print()

correlated_features = get_corr_features(df_train, 'response', 0.5)
print(correlated_features)

df_train = df_train[correlated_features]

selected_features = vif_feature_selection(df_train, 10)
print(selected_features)

df_train = selected_features


################################################################################
# DATA PRE-PROCESSING
################################################################################

X_train, y_train = separate_xy(df_train, 'response')
#X_train = min_max_normalization(X_train)

################################################################################
# REGRESSION MODELS
################################################################################

print()
print('#######################################################################')
print('LINEAR REGRESSION')
print('#######################################################################')
print()

linear_reg = LinearRegression(log_loss, X_train, y_train, max_iter=500)
linear_reg.fit()
preds = linear_reg.predict(X_train)
print("Linear regression beta vector: ")
print(linear_reg.beta)
print()
print("Linear regression loss: ")
print(log_loss(preds, y_train))

print()
print('#######################################################################')
print('RIDGE REGRESSION')
print('#######################################################################')
print()

ridge_reg = RidgeRegression(log_loss, X_train, y_train, max_iter=500, 
    regularization=0.01)
ridge_reg.fit()
preds = ridge_reg.predict(X_train)
print("Ridge regression beta vector: ")
print(ridge_reg.beta)
print()
print("Ridge regression loss: ")
print(log_loss(preds, y_train))

print()
print('#######################################################################')
print('LASSO REGRESSION')
print('#######################################################################')
print()

lasso_reg = LassoRegression(log_loss, X_train, y_train, max_iter=500, 
    regularization=0.001)
lasso_reg.fit()
y_pred = lasso_reg.predict(X_train)
print("Lasso regression beta vector: ")
print(lasso_reg.beta)
print()
print("Lasso regression loss: ")
print(log_loss(y_pred, y_train))
print()

print("Predictions:")
print(y_pred)

print("Actual values:")
print(y_train)
