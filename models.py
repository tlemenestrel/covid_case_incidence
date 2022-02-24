from scipy.optimize import minimize

# Special loss function
# formula for the loss function
# | log(1 + y) − log(1 + yˆ)|
def log_loss(y_pred, y_true):

    y_true =  np.array(y_true)
    y_pred =  np.array(y_pred)
    assert(len(y_true) == len(y_pred))

    loss = 0
    for pred, true in zip(y_pred, y_true):
        loss += abs(log(1 + true) - log(1 + pred))

    return loss

class LinearRegression:

    def __init__(self, loss_function, X, y):

        self.loss_function = loss_function
        self.X = X
        self.y = y
        self.beta = None
        self.beta_init = None

    def predict(self, X):

        # Assert the model has been fit already
        assert(type(self.beta_init) != None)

        predictions = X @ self.beta
        return predictions

    def model_error(self):

        # Assert the model has been fit already
        assert(type(self.beta_init) != None)

        error = self.loss_function(self.predict(self.X), self.y)
        return error

    def fit(self):

        # Check if the beta vector is none
        if(type(self.beta_init) == None):
            self.beta_init = np.array([1]*self.X.shape[1])
        else:
            pass

        if(self.beta != None and all(self.beta_init == self.beta)):
            print("Model has already been fit once. Refitting...")

        res = minimize(self.model_error(), self.beta_init, method='BFGS') 
        self.beta = res.x
        self.beta_init = self.beta







        



