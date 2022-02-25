# Libraries
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Code
from eda import pearson_corr_mat, get_corr_features
from models import LinearRegression, RidgeRegression, LassoRegression, log_loss
from utils import separate_xy, min_max_normalization, mean_normalization

################################################################################
# DATA PRE-PROCESSING
################################################################################

# Read training data and split in X and y
df_train = pd.read_csv('train_data.csv')

# Drop date 
df_train = df_train.drop('date', axis=1)

# Select a subset of highly correlated-features
#relevant_features = get_corr_features(df_train, 'response', 0.50)
#df_train =  df_train[relevant_features]

X_train, y_train = separate_xy(df_train, 'response')

################################################################################
# FEATURE SELECTION
################################################################################

vif = pd.DataFrame()
vif["features"] = df_train.columns
vif["vif_Factor"] = [variance_inflation_factor(df_train.values, i) for i in range(df_train.shape[1])]
print(vif)

# Select a subset of highly correlated-features
relevant_features = get_corr_features(df_train, 'response', 0.50)
df_train =  df_train[relevant_features]

vif = pd.DataFrame()
vif["features"] = df_train.columns
vif["vif_Factor"] = [variance_inflation_factor(df_train.values, i) for i in range(df_train.shape[1])]
print(vif)

################################################################################
# REGRESSION MODELS
################################################################################

linear_reg = LinearRegression(log_loss, X_train, y_train, max_iter=500)
linear_reg.fit()
preds = linear_reg.predict(X_train)
print("Linear regression beta vector: ")
print(linear_reg.beta)
print("Linear regression loss: ")
print(log_loss(preds, y_train))

ridge_reg = RidgeRegression(log_loss, X_train, y_train, max_iter=500, 
    regularization=0.0001)
ridge_reg.fit()
preds = ridge_reg.predict(X_train)
print("Ridge regression beta vector: ")
print(ridge_reg.beta)
print("Ridge regression loss: ")
print(log_loss(preds, y_train))

lasso_reg = LassoRegression(log_loss, X_train, y_train, max_iter=500, 
    regularization=0.0001)
lasso_reg.fit()
preds = lasso_reg.predict(X_train)
print("Lasso regression beta vector: ")
print(lasso_reg.beta)
print("Lasso regression loss: ")
print(log_loss(preds, y_train))





