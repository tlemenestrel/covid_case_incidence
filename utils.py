import numpy as np
import pandas as pd

def separate_xy(df, last_col_name):

    X = df.drop([last_col_name], axis=1)
    y = df[last_col_name]
    return X, y

