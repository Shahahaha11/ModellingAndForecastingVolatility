# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def metrics(y_true, y_pred, eps=1e-10):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = min(len(y_true), len(y_pred))
    y_true = np.clip(y_true[-m:], eps, None)
    y_pred = np.clip(y_pred[-m:], eps, None)

    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    nll  = 0.5 * np.mean(np.log(2*np.pi) + np.log(y_pred) + y_true / y_pred)
    qlike = np.mean(np.log(y_pred) + y_true / y_pred)
    bias = float(np.mean(y_true - y_pred))
    return {"RMSE": rmse, "MAE": mae, "NLL": nll, "QLIKE": qlike, "Bias": bias}

def metrics_new(y_true, y_pred, eps=1e-10):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)

    m = min(len(y_true), len(y_pred))
    y_true = np.clip(y_true[-m:], eps, None)
    y_pred = np.clip(y_pred[-m:], eps, None)

    # Errors
    err = y_true - y_pred

    # Core
    mse  = np.mean(err**2)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(err))
    bias = np.mean(err)

    # Likelihood / variance metrics
    qlike = np.mean(np.log(y_pred) + y_true / y_pred)
    nll   = 0.5 * np.mean(np.log(2*np.pi) + np.log(y_pred) + y_true / y_pred)

    # Relative metrics
    hmse = np.mean((1 - y_true / y_pred)**2)
    hmae = np.mean(np.abs(1 - y_true / y_pred))

    # R2
    r2 = 1 - np.sum(err**2) / np.sum((y_true - np.mean(y_true))**2)

    # MADL (variance -> signal)
    signal = y_pred - np.mean(y_pred)
    direction = np.sign(signal)
    madl = np.mean(-np.sign(y_true * direction) * np.abs(y_true))

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MSE": mse,
        "NLL": nll,
        "QLIKE": qlike,
        "HMSE": hmse,
        "HMAE": hmae,
        "R2": r2,
        "Bias": bias,
        "MADL": madl
    }