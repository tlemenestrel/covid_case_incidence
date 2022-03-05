# Libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 

# Code
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from models import log_loss
from utils import separate_xy, vif_feature_selection, get_corr_features, train_test_split, add_one_hot_and_interactions

################################################################################
# DATA PRE-PROCESSING
################################################################################

# Read the dataset
df = pd.read_csv('original_train_data.csv')

# Get the list of the columns of the dataframe
column_list = df.columns.values.tolist()
column_list.remove('Unnamed: 0')
column_list.remove('date')
column_list.remove('county')
column_list.remove('response')

# Get the shifted features
for column_name in column_list:

    df['SHIFT_' + column_name] = df[column_name] - df[column_name].shift(1) / df[column_name]
    df = df.drop(column_name, axis = 1)
    max_value = df['SHIFT_' + column_name].max()
    min_value  = df['SHIFT_' + column_name].min()
    df['SHIFT_' + column_name] = (df['SHIFT_' + column_name] - min_value) / (max_value - min_value)

df = df.dropna(axis=1, how='all')
df = df.dropna()
# df = df[df.county != 1073]

print(df.head())
df.to_csv('df.csv')

# Get train data and validation data
df_train, df_val = train_test_split(df)

print(df_train.head())
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

print(county_list_train)
print(county_list_val)
assert(len(county_list_train) == len(county_list_val))
assert(set(county_list_train)  == set(county_list_val))

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

print(df_list_train[0].head())
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

def run_models(X_train, y_train, X_val, y_val, verbose=True, cutoff_at_zero=False):
    ################################################################################
    # REGRESSION MODELS 
    ################################################################################

    if verbose:
        print()
        print('#######################################################################')
        print('LINEAR REGRESSION')
        print('#######################################################################')
        print()

    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)
    y_pred_linear = linear_reg.predict(X_val)
    if cutoff_at_zero:
        y_pred_linear[y_pred_linear < 0] = 0

    ols_loss = log_loss(y_pred_linear, y_val)
    if verbose:
        print("Linear regression loss: ")
        print(ols_loss)

        print()
        print('#######################################################################')
        print('RIDGE REGRESSION')
        print('#######################################################################')
        print()

    ridge_reg = Ridge(alpha=0.1)
    ridge_reg.fit(X_train, y_train)
    y_pred_ridge = ridge_reg.predict(X_val)
    if cutoff_at_zero:
       y_pred_ridge[y_pred_ridge < 0] = 0

    ridge_loss = log_loss(y_pred_ridge, y_val)
    if verbose:
        print("Ridge regression loss: ")
        print(ridge_loss)

        print()
        print('#######################################################################')
        print('LASSO REGRESSION')
        print('#######################################################################')
        print()

    # Increasing default tolerance so the solver converges
    lasso_reg = Lasso(alpha=0.02, tol=0.1)
    lasso_reg.fit(X_train, y_train)
    y_pred_lasso = lasso_reg.predict(X_val)
    if cutoff_at_zero:
       y_pred_lasso[y_pred_lasso < 0] = 0

    lasso_loss = log_loss(y_pred_lasso, y_val)
    if verbose:
        print("Lasso regression loss: ")
        print(lasso_loss)
        print()

        print("Predictions:")
        print(y_pred_lasso)

        print("Actual values:")
        print(y_val)

    return (ols_loss, ridge_loss, lasso_loss)

run_models(X_train, y_train, X_val, y_val)