# Libraries
import pandas as pd
pd.options.mode.chained_assignment = None 
from math import log
import numpy as np
from scipy.optimize import minimize

# Code
from sklearn.linear_model import LinearRegression, Lasso, Ridge

# Special loss function
# formula for the loss function
# |log(1 + y) − log(1 + yˆ)|
def log_loss(y_pred, y_true):

    y_true =  np.array(y_true)
    y_pred =  np.array(y_pred)
    assert(len(y_true) == len(y_pred))

    return np.mean(np.abs(np.log(1 + y_true) - np.log(1 + y_pred)))


def l1_loss(y_pred, y_true):
    y_true =  np.array(y_true)
    y_pred =  np.array(y_pred)
    assert(len(y_true) == len(y_pred))

    return np.mean(np.abs(y_true -y_pred))


class CustomLinearRegression:

    def __init__(self, loss_function, X, y, max_iter):

        self.loss_function = loss_function
        self.X = X
        self.y = y
        self.beta = None
        self.beta_init = None
        self.max_iter = max_iter

    def predict(self, X):

        # Assert the model has been fit already
        assert(type(self.beta_init) != None)
        predictions = X @ self.beta
        return predictions

    def model_error(self):

        error = self.loss_function(self.predict(self.X), self.y)
        return(error)

    def objective_function(self, beta):
        self.beta = beta
        return self.model_error()

    def fit(self):

        # Check if the beta vector is none
        if(type(self.beta_init) == type(None)):
            self.beta_init = np.ones(self.X.shape[1])
        else:
            pass

        if(self.beta != None and all(self.beta_init == self.beta)):
            print("Model has already been fit once. Refitting...")

        res=minimize(self.objective_function,self.beta_init,method='BFGS', 
            options={'maxiter': self.max_iter}) 
        self.beta = res.x
        self.beta_init = self.beta

class RidgeRegression:

    def __init__(self, loss_function, X, y, max_iter, regularization):

        self.loss_function = loss_function
        self.X = X
        self.y = y
        self.beta = None
        self.beta_init = None
        self.max_iter = max_iter
        self.regularization = regularization

    def predict(self, X):

        # Assert the model has been fit already
        assert(type(self.beta_init) != None)
        predictions = X @ self.beta
        return predictions

    def model_error(self):

        error = self.loss_function(self.predict(self.X), self.y)
        return(error)

    def l2_loss(self, beta):
        self.beta = beta
        return (self.model_error()+sum(self.regularization*np.array(self.beta)**2))

    def fit(self):

        # Check if the beta vector is none
        if(type(self.beta_init) == type(None)):
            self.beta_init = np.ones(self.X.shape[1])
        else:
            pass

        if(self.beta != None and all(self.beta_init == self.beta)):
            print("Model has already been fit once. Refitting...")

        res=minimize(self.l2_loss, self.beta_init, method='BFGS', 
            options={'maxiter': self.max_iter}) 
        self.beta = res.x
        self.beta_init = self.beta

class LassoRegression:

    def __init__(self, loss_function, X, y, max_iter, regularization):

        self.loss_function = loss_function
        self.X = X
        self.y = y
        self.beta = None
        self.beta_init = None
        self.max_iter = max_iter
        self.regularization = regularization

    def predict(self, X):

        # Assert the model has been fit already
        assert(type(self.beta_init) != None)
        predictions = X @ self.beta
        return predictions

    def model_error(self):

        error = self.loss_function(self.predict(self.X), self.y)
        return(error)

    def l2_loss(self, beta):
        self.beta = beta
        return (self.model_error()+sum(self.regularization*np.array(self.beta)))

    def fit(self):

        # Check if the beta vector is none
        if(type(self.beta_init) == type(None)):
            self.beta_init = np.ones(self.X.shape[1])
        else:
            pass

        if(self.beta != None and all(self.beta_init == self.beta)):
            print("Model has already been fit once. Refitting...")

        res=minimize(self.l2_loss, self.beta_init, method='BFGS', 
            options={'maxiter': self.max_iter}) 
        self.beta = res.x
        self.beta_init = self.beta



def run_models(X_train, y_train, X_val, y_val, verbose=True, cutoff_at_zero=False, lasso_alpha=0.02):
    ################################################################################
    # REGRESSION MODELS 
    ################################################################################

    if verbose:
        print()
        print('#######################################################################')
        print('LINEAR REGRESSION')
        print('#######################################################################')
        print()

    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)
    y_pred_linear = linear_reg.predict(X_val)
    if cutoff_at_zero:
        y_pred_linear[y_pred_linear < 0] = 0

    ols_loss = log_loss(y_pred_linear, y_val)
    if verbose:
        print("Linear regression loss: ")
        print(ols_loss)

        print()
        print('#######################################################################')
        print('RIDGE REGRESSION')
        print('#######################################################################')
        print()

    ridge_reg = Ridge(alpha=0.0001)
    ridge_reg.fit(X_train, y_train)
    y_pred_ridge = ridge_reg.predict(X_val)
    if cutoff_at_zero:
       y_pred_ridge[y_pred_ridge < 0] = 0

    ridge_loss = log_loss(y_pred_ridge, y_val)
    if verbose:
        print("Ridge regression loss: ")
        print(ridge_loss)

        print()
        print('#######################################################################')
        print('LASSO REGRESSION')
        print('#######################################################################')
        print()

    # Increasing default tolerance so the solver converges
    lasso_reg = Lasso(alpha=lasso_alpha, tol=0.1)
    lasso_reg.fit(X_train, y_train)
    y_pred_lasso = lasso_reg.predict(X_val)
    if cutoff_at_zero:
       y_pred_lasso[y_pred_lasso < 0] = 0

    lasso_loss = log_loss(y_pred_lasso, y_val)
    if verbose:
        print("Lasso regression loss: ")
        print(lasso_loss)
        print()

        print("Predictions:")
        print(y_pred_linear)

        print("Actual values:")
        print(y_val)

    return (ols_loss, ridge_loss, lasso_loss)
