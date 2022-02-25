from math import log
import numpy as np
from scipy.optimize import minimize

# Special loss function
# formula for the loss function
# | log(1 + y) − log(1 + yˆ)|
def log_loss(y_pred, y_true):

    y_true =  np.array(y_true)
    y_pred =  np.array(y_pred)
    assert(len(y_true) == len(y_pred))

    return np.mean(np.abs(np.log(1 + y_true) - np.log(1 + y_pred)))

class LinearRegression:

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
        print(self.beta)








