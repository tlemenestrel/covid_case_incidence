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
df_train['county'] = df_train['county'].astype(str).str[:-3].astype(np.int64)
df_train = pd.get_dummies(df_train, prefix=['county'], columns=['county'])

################################################################################
# FEATURE SELECTION
################################################################################

print()
print('#######################################################################')
print('FEATURE SELECTION')
print('#######################################################################')
print()

X_train, y_train = separate_xy(df_train, 'response')

'''
print(df_train.head())

X_train, y_train = separate_xy(df_train, 'response')

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

lasso_reg = Lasso(alpha=10)
lasso_reg.fit(X_train, y_train)
y_pred_lasso = lasso_reg.predict(X_train)

print("Lasso regression loss: ")
print(mean_squared_error(y_pred_lasso, y_train))
print()

print("Predictions:")
print(y_pred_ridge)

print("Actual values:")
print(y_train)