# Libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 

# Code
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from models import log_loss
from utils import separate_xy, vif_feature_selection, get_corr_features, train_test_split, add_one_hot_and_interactions
from main import run_models
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

# Get train data and validation data
df_train, df_val = train_test_split(df)

################################################################################
# SPLITTING DATA
################################################################################

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
ols_sum = ridge_sum = lasso_sum = 0
best_lasso_alphas = []
lasso_losses = []
lasso_alphas_candidates = np.linspace(0, 1, 501)
for train, val in zip(df_list_train, df_list_val):
    train = train.drop('date', axis=1)
    val   = val.drop('date', axis=1)

    X_train, y_train = separate_xy(train, 'response')
    X_val, y_val     = separate_xy(val, 'response')

    min_lasso = 1000
    best_alpha = 0
    for alpha in lasso_alphas_candidates[1:]:
        ols, ridge, lasso = run_models(X_train, y_train, X_val, y_val, verbose=False, cutoff_at_zero=True, lasso_alpha=alpha)
        if lasso < min_lasso:
            min_lasso = lasso
            best_alpha = alpha
    
    ols_sum += ols
    ridge_sum += ridge
    lasso_sum += min_lasso
    best_lasso_alphas.append(best_alpha)
    lasso_losses.append(min_lasso)

ols_sum /= len(df_list_train)
ridge_sum /= len(df_list_train)
lasso_sum /= len(df_list_train)

print(ols_sum, ridge_sum, lasso_sum)

print(best_lasso_alphas)

print(lasso_losses)