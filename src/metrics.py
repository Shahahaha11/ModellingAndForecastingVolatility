# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import numpy as np

# def metrics(y_true, y_pred, eps=1e-10):
#     y_true = np.asarray(y_true, float)
#     y_pred = np.asarray(y_pred, float)
#     m = min(len(y_true), len(y_pred))
#     y_true, y_pred = y_true[-m:], np.clip(y_pred[-m:], eps, None)

#     mse  = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     mae  = mean_absolute_error(y_true, y_pred)
#     nll  = 0.5 * np.mean(np.log(2*np.pi*y_pred) + y_true / y_pred)
#     qlike = np.mean(y_true / y_pred - np.log(y_true / y_pred + eps) - 1.0)
#     bias = float(np.mean(y_true - y_pred))
#     return {"RMSE": rmse, "MAE": mae, "NLL": nll, "QLIKE": qlike, "Bias": bias}


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
