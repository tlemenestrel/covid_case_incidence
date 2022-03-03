# Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

# Code
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from models import log_loss
from utils import separate_xy, min_max_normalization, mean_normalization, vif_feature_selection, get_corr_features

################################################################################
# DATA PRE-PROCESSING
################################################################################

# Read train data
df_train = pd.read_csv('temp_train.csv')
save_train = df_train[['date', 'county']]
df_train = df_train.drop('date', axis=1)
df_train = df_train.drop('county', axis=1)

# Keep month and day and drop date
save_train["date"]  = pd.to_datetime(save_train["date"])
save_train["month"] = save_train['date'].dt.month
save_train["day"]   = save_train['date'].dt.month
save_train = save_train.drop('date', axis=1)

# Counties
#df_train['county'] = df_train['county'].astype(str).str[:-3].astype(np.int64)
save_train = pd.get_dummies(save_train, prefix=['county'], columns=['county'])

# Read validation data
df_val  = pd.read_csv('temp_val.csv')

save_val = df_val[['date', 'county']]
df_val = df_val.drop('date', axis=1)
df_val = df_val.drop('county', axis=1)

# Keep month and day and drop date
save_val["date"]  = pd.to_datetime(save_val["date"])
save_val["month"] = save_val['date'].dt.month
save_val["day"]   = save_val['date'].dt.month
save_val = save_val.drop('date', axis=1)

# Counties
#df_val['county'] = df_val['county'].astype(str).str[:-3].astype(np.int64)
save_val = pd.get_dummies(save_val, prefix=['county'], columns=['county'])

################################################################################
# FEATURE SELECTION
################################################################################

print()
print('#######################################################################')
print('FEATURE SELECTION')
print('#######################################################################')
print()

X_train, y_train = separate_xy(df_train, 'response')
#X_train = mean_normalization(X_train)

X_val, y_val = separate_xy(df_val, 'response')
#X_val = mean_normalization(X_val)

# Select features based on a correlation threshold
correlated_features = get_corr_features(df_train, 'response', 0.3)
df_train = df_train[correlated_features]

print("Correlated features:")
print(correlated_features)

# Split data into x and y
X_train, y_train = separate_xy(df_train, 'response')

# Select features based on VIF analysis
selected_features = vif_feature_selection(X_train, 10)
X_train = X_train[selected_features]
X_val   = X_val[selected_features]
print("VIF selected features:")
print(selected_features)

# Combine dataframes
X_train = pd.concat([X_train, save_train], axis=1)
X_val   = pd.concat([X_val, save_val], axis=1)

print(X_train.head())

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
y_pred_linear = linear_reg.predict(X_val)

print("Linear regression loss: ")
print(log_loss(y_pred_linear, y_val))

print()
print('#######################################################################')
print('RIDGE REGRESSION')
print('#######################################################################')
print()

ridge_reg = Ridge(alpha=1)
ridge_reg.fit(X_train, y_train)
y_pred_ridge = ridge_reg.predict(X_val)

print("Ridge regression loss: ")
print(log_loss(y_pred_ridge, y_val))

print()
print('#######################################################################')
print('LASSO REGRESSION')
print('#######################################################################')
print()

lasso_reg = Lasso(alpha=1)
lasso_reg.fit(X_train, y_train)
y_pred_lasso = lasso_reg.predict(X_val)

print("Lasso regression loss: ")
print(log_loss(y_pred_lasso, y_val))
print()

print("Predictions:")
print(y_pred_ridge)

print("Actual values:")
print(y_val)