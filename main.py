# Libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 

from utils import separate_xy, vif_feature_selection, get_corr_features, train_test_split, \
    add_one_hot_and_interactions, add_shifted_features
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from models import log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from time_based_cv import TimeBasedCV
import pandas as pd
import datetime
from datetime import datetime as dt
from dateutil.relativedelta import *
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=UserWarning)


################################################################################
# DATA PRE-PROCESSING
################################################################################

# Read the dataset
df = pd.read_csv('original_train_data.csv')
df['date'] = pd.to_datetime(df['date'])

X, y  = separate_xy(df, 'response')

################################################################################
# PARAMETERS FOR CV
################################################################################

tscv = TimeBasedCV(train_period=30,
                   test_period=7,
                   freq='days')

split_date = datetime.date(2020, 11, 1)
for train_index, test_index in tscv.split(df, validation_split_date=split_date, date_column='date'):
    continue

# get number of splits
tscv.get_n_splits()
print(tscv.get_n_splits())

# Make a list to store all the alphas
alphas = []
for alpha in np.linspace(0, 15, 300):
    alphas.append(alpha)

# Make a dictionnary of lists to store the CV scores for each alpha and for each fold
alpha_dict = { i : [] for i in alphas}

################################################################################
# TIME-BASED CV
################################################################################

for train_index, test_index in tscv.split(X, validation_split_date=datetime.date(2020, 11, 1)):

    # Split the data
    X_train   = X.loc[train_index].drop('date', axis=1)
    y_train = y.loc[train_index]

    X_test    = X.loc[test_index].drop('date', axis=1)
    y_test  = y.loc[test_index]

    # For each value of the hyperparamter alpha
    for alpha in alphas:
        # Make model, fit and make predictions
        model = make_pipeline(StandardScaler(), ElasticNet(random_state=0, alpha=alpha, l1_ratio=1))
        y_pred = model.fit(X_train, y_train).predict(X_test)

        # Compute loss
        loss = log_loss(y_pred, y_test)

        # Add the loss to the corresponding alpha list in the dictionnary
        alpha_dict[alpha].append(loss)

# Compute the average loss for each alpha in the dictionnary
for key in alpha_dict.keys():
    alpha_dict[key] = np.mean(alpha_dict[key])

# Compute t
lowest_average_loss = min(alpha_dict.values())
best_alpha = [k for k, v in alpha_dict.items() if v == lowest_average_loss]
print("Lowest average loss after CV: " + str(lowest_average_loss))
print("Best alpha after CV: " + str(best_alpha))

# Get train data and validation data
df_train, df_val = train_test_split(df)

df_train = df_train.drop('date', axis=1)

df_val = df_val.drop('date', axis=1)

print(df_train.head())

X_train, y_train = separate_xy(df_train, 'response')
X_val, y_val     = separate_xy(df_val, 'response')

model = make_pipeline(StandardScaler(), ElasticNet(random_state=0, alpha=best_alpha, l1_ratio=1))

model.fit(X_train, y_train)
y_pred = model.predict(X_val)
lasso_loss = log_loss(y_pred, y_val)

print(lasso_loss)
