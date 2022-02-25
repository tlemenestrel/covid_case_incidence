import numpy as np
import pandas as pd

def separate_xy(df, last_col_name):

    X = df.drop([last_col_name], axis=1)
    y = df[last_col_name]
    return X, y

def mean_normalization(df):

    result = df.copy()
    for feature_name in df.columns:
        mean_value = df[feature_name].mean()
        std_value  = df[feature_name].std()
        result[feature_name] = (df[feature_name] - mean_value) / std_value
    return result

def min_max_normalization(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

