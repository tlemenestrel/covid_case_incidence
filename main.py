# Libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 

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

print(df.head())
df.to_csv('df.csv')

# Get train data and validation data
df_train, df_val = train_test_split(df)

print(df_train.head())
################################################################################
# SPLITTING DATA
################################################################################

# Make lists to store the sub-dataframes
df_list_train = []
df_list_val   = []

# Slice the train and val dataframe into lists of sub dataframes for each county.
for county_code in county_list:

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

run_models(X_train, y_train, X_val, y_val, cutoff_at_zero=True)