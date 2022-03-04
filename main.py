# Libraries
import numpy as np
import pandas as pd

# Code
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from models import log_loss
from utils import separate_xy, vif_feature_selection, get_corr_features

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
df_train['county'] = df_train['county'].apply(str).str.zfill(5)
df_val['county']   =   df_val['county'].apply(str).str.zfill(5)

# Get the list of unique counties
county_list_train = df_train['county'].unique()
county_list_val   = df_val['county'].unique()

# Check that the counties are the same and in the same order
assert(len(county_list_train) == len(county_list_val))
assert(all(county_list_train  == county_list_val))

# Make lists to store the sub-dataframes
df_list_train = []
df_list_val   = []

# Slice the train and val dataframe into lists of sub dataframes for each county.
for county_code in county_list_train:

    train_sub_df = df_train.loc[df_train['county'] == county_code]
    df_list_train.append(train_sub_df)

    val_sub_df   = df_val.loc[df_val['county'] == county_code]
    df_list_val.append(val_sub_df)

# Try to fit a separate model on the data for the first county
df_list_train[0] = df_list_train[0].drop('date', axis=1)
df_list_val[0]   = df_list_val[0].drop('date', axis=1)

'''
correlated_features = get_corr_features(df_list_train[0], 'response', 0.5)
df_list_train[0] = df_list_train[0][correlated_features]
df_list_val[0]   = df_list_val[0][correlated_features]
print(correlated_features)
'''

X_train, y_train = separate_xy(df_list_train[0], 'response')
X_val, y_val     = separate_xy(df_list_val[0], 'response')

'''
selected_features = vif_feature_selection(X_train, 10)
X_train = X_train[selected_features]
X_val   = X_val[selected_features]
print(selected_features)
'''

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

ridge_reg = Ridge(alpha=0.005)
ridge_reg.fit(X_train, y_train)
y_pred_ridge = ridge_reg.predict(X_val)

print("Ridge regression loss: ")
print(log_loss(y_pred_ridge, y_val))

print()
print('#######################################################################')
print('LASSO REGRESSION')
print('#######################################################################')
print()

lasso_reg = Lasso(alpha=100)
lasso_reg.fit(X_train, y_train)
y_pred_lasso = lasso_reg.predict(X_val)

print("Lasso regression loss: ")
print(log_loss(y_pred_lasso, y_val))
print()

print("Predictions:")
print(y_pred_ridge)

print("Actual values:")
print(y_val)
