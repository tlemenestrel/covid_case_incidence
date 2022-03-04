# Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

# Code
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from models import log_loss
from utils import separate_xy, min_max_normalization, mean_normalization, train_test_split

################################################################################
# DATA PRE-PROCESSING
################################################################################

# Read train data
df_train = pd.read_csv('train_data.csv')

# Read validation data
df_val  = pd.read_csv('val_data.csv')

################################################################################
# SPLITTING DATA
################################################################################

# Make column to string and add leading zeroes that are removed when reading file
df_train["county"] = df_train["county"].apply(str).str.zfill(5)
df_val["county"]   = df_val["county"].apply(str).str.zfill(5)

county_list_train = df_train['county'].unique()
county_list_val   = df_val['county'].unique()

assert(len(county_list_train) == len(county_list_val))
assert(sorted(county_list_train)  == sorted(county_list_val))
print(df_train.head())
print(sorted(county_list_train))
print(sorted(county_list_val))

################################################################################
# REGRESSION MODELS 
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

ridge_reg = Ridge(alpha=10)
ridge_reg.fit(X_train, y_train)
y_pred_ridge = ridge_reg.predict(X_val)

print("Ridge regression loss: ")
print(log_loss(y_pred_ridge, y_val))

print()
print('#######################################################################')
print('LASSO REGRESSION')
print('#######################################################################')
print()

lasso_reg = Lasso(alpha=0.00001)
lasso_reg.fit(X_train, y_train)
y_pred_lasso = lasso_reg.predict(X_val)

print("Lasso regression loss: ")
print(log_loss(y_pred_lasso, y_val))
print()

print("Predictions:")
print(y_pred_ridge)

print("Actual values:")
print(y_val)
