# Libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 

# Code
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from models import log_loss
from utils import separate_xy, vif_feature_selection, get_corr_features, train_test_split, \
    add_one_hot_and_interactions, add_shifted_features
from models import run_models

################################################################################
# DATA PRE-PROCESSING
################################################################################

# Read the dataset
df = pd.read_csv('original_train_data.csv')

# Make column to string and add leading zeroes that are removed when reading file
df['county'] = df['county'].apply(str).str.zfill(5)

# Get the list of unique counties
county_list = df['county'].unique().tolist()

# Get the list of the columns of the dataframe
column_list = df.columns.values.tolist()
column_list.remove('Unnamed: 0')
column_list.remove('date')
column_list.remove('county')
column_list.remove('response')

# Add the shifted features
df = add_shifted_features(df, county_list, column_list)

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
lasso_losses = []

from constant import best_lasso_alphas
i = 0
for train, val in zip(df_list_train, df_list_val):
    train = train.drop('date', axis=1)
    val   = val.drop('date', axis=1)

    X_train, y_train = separate_xy(train, 'response')
    X_val, y_val     = separate_xy(val, 'response')

    alpha = best_lasso_alphas[i]
    ols, ridge, lasso = run_models(X_train, y_train, X_val, y_val, verbose=False, cutoff_at_zero=True, lasso_alpha=alpha)
    
    ols_sum += ols
    ridge_sum += ridge
    lasso_sum += lasso
    lasso_losses.append(lasso)
    i += 1

ols_sum /= len(df_list_train)
ridge_sum /= len(df_list_train)
lasso_sum /= len(df_list_train)

print(ols_sum, ridge_sum, lasso_sum)

print(lasso_losses)