from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

import numpy as np

def variance_metrics(y_true, y_pred, eps=1e-10):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("Invalid true and forecast arrays")

    err = y_true - y_pred

    true_pos = np.clip(y_true, eps, None)
    pred_pos = np.clip(y_pred, eps, None)

    return {
        "RMSE": np.sqrt(np.mean(err**2)),
        "MAE": np.mean(np.abs(err)),
        "HMSE": np.mean((1 - true_pos / pred_pos)**2),
        "HMAE": np.mean(np.abs(1 - true_pos / pred_pos)),
        "QLIKE": np.mean(np.log(pred_pos) + true_pos / pred_pos),
        "NLL": 0.5 * np.mean(
            np.log(2 * np.pi) + np.log(pred_pos) + true_pos / pred_pos
        ),
        "Bias": np.mean(err)
    }