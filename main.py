# Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

# Code
from eda import pearson_corr_mat, get_corr_features
from models import LinearRegression, log_loss
from utils import separate_xy, min_max_normalization

################################################################################
# DATA PRE-PROCESSING
################################################################################

# Read training data and split in X and y
df_train = pd.read_csv('train_data.csv')

# Drop date 
df_train = df_train.drop('date', axis=1)

# Select a subset of highly correlated-features
relevant_features = get_corr_features(df_train, 'response', 0.65)
df_train =  df_train[relevant_features]

X_train, y_train = separate_xy(df_train, 'response')
X_train = min_max_normalization(X_train)

print(X_train)
print(y_train)

################################################################################
# LINEAR REGRESSION
################################################################################

linear_reg = LinearRegression(log_loss, X_train, y_train, max_iter=1000)
linear_reg.fit()
preds = linear_reg.predict(X_train)
print(log_loss(preds, y_train))

print(preds)
print(y_train)