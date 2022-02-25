# Libraries
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Code
from eda import pearson_corr_mat, get_corr_features
from models import LinearRegression, RidgeRegression, LassoRegression, log_loss
from utils import separate_xy, min_max_normalization, mean_normalization

################################################################################
# DATA PRE-PROCESSING
################################################################################

# Read training data, split in X and y and drop date column
df_train = pd.read_csv('train_data.csv')
df_train = df_train.drop('date', axis=1)

################################################################################
# FEATURE SELECTION
################################################################################

# Select a subset of highly correlated-features
relevant_features = [ 

'fb-survey_smoothed_cli',
'chng_smoothed_outpatient_covid',
'hospital-admissions_smoothed_adj_covid19_from_claims',
'safegraph_part_time_work_prop_7dav',
'doctor-visits_smoothed_adj_cli',
'quidel_covid_ag_smoothed_pct_positive',
'response',
'fb-survey_smoothed_wtravel_outside_state_5d'

]

df_train =  df_train[relevant_features]

print()
print('#######################################################################')
print('VIF ANALYSIS')
print('#######################################################################')
print()

# VIF analysis to check for multicollinearity
vif = pd.DataFrame()
vif["features"] = df_train.columns
vif["vif_Factor"] = [variance_inflation_factor(df_train.values, i) for i in range(df_train.shape[1])]
print(vif)

################################################################################
# REGRESSION MODELS
################################################################################

X_train, y_train = separate_xy(df_train, 'response')

print()
print('#######################################################################')
print('LINEAR REGRESSION')
print('#######################################################################')
print()

linear_reg = LinearRegression(log_loss, X_train, y_train, max_iter=500)
linear_reg.fit()
preds = linear_reg.predict(X_train)
print("Linear regression beta vector: ")
print(linear_reg.beta)
print()
print("Linear regression loss: ")
print(log_loss(preds, y_train))

print()
print('#######################################################################')
print('RIDGE REGRESSION')
print('#######################################################################')
print()

ridge_reg = RidgeRegression(log_loss, X_train, y_train, max_iter=500, 
    regularization=0.0001)
ridge_reg.fit()
preds = ridge_reg.predict(X_train)
print("Ridge regression beta vector: ")
print(ridge_reg.beta)
print()
print("Ridge regression loss: ")
print(log_loss(preds, y_train))

print()
print('#######################################################################')
print('LASSO REGRESSION')
print('#######################################################################')
print()

lasso_reg = LassoRegression(log_loss, X_train, y_train, max_iter=500, 
    regularization=0.0001)
lasso_reg.fit()
preds = lasso_reg.predict(X_train)
print("Lasso regression beta vector: ")
print(lasso_reg.beta)
print()
print("Lasso regression loss: ")
print(log_loss(preds, y_train))
print()
