

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
