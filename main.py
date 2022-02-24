import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from models import LinearRegression, log_loss
from utils import separate_xy

# Read training data and split in X and y
df_train = pd.read_csv('train_data.csv')
X_train, y_train = separate_xy(df_train, 'response')

print(df_train.head())
print(X_train.head())
print(y_train.head())

linear_reg = LinearRegression(log_loss, X_train, y_train)
linear_reg.fit()
preds = linear_reg.predict(X_train)
print(log_loss(preds, y_train))