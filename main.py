# Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

# Code
from sklearn.linear_model import LinearRegression, Lasso, Ridge
#from models import LinearRegression, RidgeRegression, LassoRegression, log_loss
from models import log_loss
from utils import separate_xy, min_max_normalization, mean_normalization, vif_feature_selection, get_corr_features

################################################################################
# DATA PRE-PROCESSING
################################################################################

df_train = pd.read_csv('train_data.csv')

# Keep month and day and drop date
df_train["date"]  = pd.to_datetime(df_train["date"])
df_train["month"] = df_train['date'].dt.month
df_train["day"]   = df_train['date'].dt.month
df_train = df_train.drop('date', axis=1)

################################################################################
# FEATURE SELECTION
################################################################################

print()
print('#######################################################################')
print('FEATURE SELECTION')
print('#######################################################################')
print()

# Select features based on a correlation threshold
correlated_features = get_corr_features(df_train, 'response', 0.55)
df_train = df_train[correlated_features]

print("Correlated features:")
print(correlated_features)

# Split data into x and y
X_train, y_train = separate_xy(df_train, 'response')

# Select features based on VIF analysis
selected_features = vif_feature_selection(X_train, 6)
X_train = selected_features

print("VIF selected features:")
print(selected_features)

################################################################################
# REGRESSION MODELS BY HAND
################################################################################

'''
print()
print('#######################################################################')
print('LINEAR REGRESSION')
print('#######################################################################')
print()

linear_reg = LinearRegression(log_loss, X_train, y_train, max_iter=500)
linear_reg.fit()
y_pred_linear = linear_reg.predict(X_train)
print("Linear regression beta vector: ")
print(linear_reg.beta)
print()
print("Linear regression loss: ")
print(log_loss(y_pred_linear, y_train))

print()
print('#######################################################################')
print('RIDGE REGRESSION')
print('#######################################################################')
print()

ridge_reg = RidgeRegression(log_loss, X_train, y_train, max_iter=500, 
    regularization=0.1)
ridge_reg.fit()
y_pred_ridge = ridge_reg.predict(X_train)
print("Ridge regression beta vector: ")
print(ridge_reg.beta)
print()
print("Ridge regression loss: ")
print(log_loss(y_pred_ridge, y_train))

print()
print('#######################################################################')
print('LASSO REGRESSION')
print('#######################################################################')
print()

lasso_reg = LassoRegression(log_loss, X_train, y_train, max_iter=500, 
    regularization=0.1)
lasso_reg.fit()
y_pred_lasso = lasso_reg.predict(X_train)
print("Lasso regression beta vector: ")
print(lasso_reg.beta)
print()
print("Lasso regression loss: ")
print(log_loss(y_pred_lasso, y_train))
print()

print("Predictions:")
print(y_pred_linear)

print("Actual values:")
print(y_train)
'''

################################################################################
# REGRESSION MODELS SKLEARN
################################################################################

print()
print('#######################################################################')
print('LINEAR REGRESSION')
print('#######################################################################')
print()

linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred_linear = linear_reg.predict(X_train)

print("Linear regression loss: ")
print(mean_squared_error(y_pred_linear, y_train))

print()
print('#######################################################################')
print('RIDGE REGRESSION')
print('#######################################################################')
print()

ridge_reg = Ridge(alpha=1)
ridge_reg.fit(X_train, y_train)
y_pred_ridge = ridge_reg.predict(X_train)

print("Ridge regression loss: ")
print(mean_squared_error(y_pred_ridge, y_train))

print()
print('#######################################################################')
print('LASSO REGRESSION')
print('#######################################################################')
print()

lasso_reg = Lasso(alpha=1)
lasso_reg.fit(X_train, y_train)
y_pred_lasso = lasso_reg.predict(X_train)

print("Lasso regression loss: ")
print(mean_squared_error(y_pred_lasso, y_train))
print()

print("Predictions:")
print(y_pred_linear)

print("Actual values:")
print(y_train)