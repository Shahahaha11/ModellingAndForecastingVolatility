import pandas as pd
from arch import arch_model
import numpy as np
import matplotlib.pyplot as plt


def estimate_naive10(Xq):
    f_rollB, y_true, idx = [], [], []
    Xn = Xq.to_numpy(dtype=float)

    for i in range(10, len(Xn)):
        r_sub = Xn[i-10:i, 0]

        f_next = r_sub.var(ddof=1)

        f_rollB.append(f_next)
        y_true.append(Xn[i, 0] ** 2)
        idx.append(i)

    return np.array(f_rollB), np.array(y_true), np.array(idx)

def estimate_garch(Xq, lookback = 122):
    f_rollB, y_true, idx = [], [], []
    Xn = Xq.to_numpy(dtype=float)

    for i in range(lookback, len(Xn)):
        r_sub = Xn[i-lookback:i, 0] * 100

        am = arch_model(r_sub, mean="Zero", vol="GARCH", p=1, q=1, rescale=False)
        res = am.fit(disp="off", show_warning=False)

        f_next = res.forecast(horizon=1).variance.values[-1, 0] / 10000

        f_rollB.append(f_next)
        y_true.append(Xn[i, 0] ** 2)
        idx.append(i)

    return np.array(f_rollB), np.array(y_true), np.array(idx)

from xgboost import XGBRegressor
import numpy as np

def estimate_xgboost(Xq_df, target_col="rv", lookback=10):
    f_rollB = []
    y_true = []
    idx = []

    for t in range(lookback, len(Xq_df)):
        train = Xq_df.iloc[t - lookback:t] # where t is not included
        test = Xq_df.iloc[[t]]

        X_train = train.drop(columns=[target_col])
        y_train = train[target_col]

        X_test = test.drop(columns=[target_col])
        y_test = test[target_col].iloc[0]

        model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            objective="reg:squarederror",
            random_state=11
        )
        model.fit(X_train, y_train)

        pred = model.predict(X_test)[0]

        f_rollB.append(pred)
        y_true.append(y_test)
        idx.append(t)

    return np.array(f_rollB), np.array(y_true), np.array(idx)

from xgboost import XGBRegressor
import numpy as np
import pandas as pd

def estimate_xgboost_par(series, lookback=10, n_lags=5, **xgb_kwargs):
    """
    Forecast a single series using its own lagged values (pure autoregressive design).
    Parameters
    ----------
    series : pd.Series or array-like
        The single time series to forecast (e.g. realized variance).
    lookback : int
        Size of the rolling training window (number of past observations
        used to fit each model).
    n_lags : int
        Number of lags of `series` used as features, i.e. x_i = (y_{t-1}, ..., y_{t-n_lags}).
    xgb_kwargs : dict
        Passed through to XGBRegressor (defaults set below if not provided).

    Returns
    -------
    f_rollB : np.ndarray of predictions
    y_true  : np.ndarray of realized values
    idx     : np.ndarray of original series indices corresponding to each forecast
    
     the forecast stored at the 20:00 index is a forecast made using information available up through 19:00,
     targeting the 20:00 value.
    """
    y = pd.Series(series).reset_index(drop=True)

    # build lag matrix from the series itself only
    df = pd.DataFrame({"y": y})
    for lag in range(1, n_lags + 1):
        df[f"lag{lag}"] = df["y"].shift(lag)
    df = df.dropna().reset_index(drop=True)  # first n_lags rows consumed

    feature_cols = [f"lag{lag}" for lag in range(1, n_lags + 1)]

    default_params = dict(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=11
    )
    default_params.update(xgb_kwargs)

    f_rollB = []
    y_true = []
    idx = []

    for t in range(lookback, len(df)):
        train = df.iloc[t - lookback:t]
        test = df.iloc[[t]]

        X_train = train[feature_cols]
        y_train = train["y"]
        X_test = test[feature_cols]
        y_test = test["y"].iloc[0]

        model = XGBRegressor(**default_params)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]

        f_rollB.append(pred)
        y_true.append(y_test)
        idx.append(t + n_lags)

    return np.array(f_rollB), np.array(y_true), np.array(idx)

import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents

def estimate_sv(Xq_df, return_col="rt", lookback=10, eps=1e-6):
    f_rollB, y_true, idx = [], [], []
    # 1. Scale returns by 100 to stabilize log calculation
    r_orig = Xq_df[return_col].astype(float).to_numpy()
    r_scaled = r_orig * 100

    for t in range(lookback, len(r_scaled)):
        # 2. Use scaled returns for the log-variance proxy
        y = np.log(r_scaled[t - lookback:t] ** 2 + eps)

        try:
            # Local level model is the standard state-space approximation for SV
            res = UnobservedComponents(y, level="local level").fit(disp=False)
            m = float(res.forecast(1)[0])
            # Forecast error covariance
            v = float(res.filter_results.forecasts_error_cov[0, 0, -1]) if res.filter_results.forecasts_error_cov.size else 0.0
            
            # 3. Exponentiate and scale back down (divide by 100^2 = 10000)
            f_next = np.exp(m + 0.5 * max(v, 0.0)) / 10000
        except Exception:
            # Robust fallback: Simple Moving Average of squared returns
            f_next = float(np.mean(r_orig[t - lookback:t] ** 2))

        f_rollB.append(f_next)
        y_true.append(r_orig[t] ** 2)
        idx.append(t)

    return np.array(f_rollB), np.array(y_true), np.array(idx)


from arch import arch_model

def estimate_garch_variant(Xq_df, return_col="rt", lookback=10, variant='GJR'):
    """
    variant: 'GJR' (for GJR-GARCH) or 'EGARCH'
    """
    f_rollB, y_true, idx = [], [], []
    r_orig = Xq_df[return_col].astype(float).to_numpy()
    
    # Scale returns by 100 for optimizer stability
    r_scaled = r_orig * 100

    for t in range(lookback, len(r_scaled)):
        window = r_scaled[t - lookback:t]
        
        try:
            if variant == 'GJR':
                # p=1, o=1, q=1 is GJR-GARCH
                model = arch_model(window, p=1, o=1, q=1, vol='Garch', dist='t')
            else:
                model = arch_model(window, p=1, o=1, q=1, vol='EGARCH', dist='t')
            
            res = model.fit(disp='off', show_batch=False)
            # Forecast 1-step ahead variance
            f_scaled = res.forecast(horizon=1).variance.values[-1, 0]
            # Scale back down (divide by 100^2)
            f_next = f_scaled / 10000
        except Exception:
            f_next = float(np.mean(r_orig[t - lookback:t] ** 2))

        f_rollB.append(f_next)
        idx.append(t)

    return np.array(f_rollB), np.array(idx)



def estimate_tarch(Xq_df, return_col="rt", lookback=10):
    f_rollB, y_true, idx = [], [], []
    r = Xq_df[return_col].astype(float).to_numpy()
    for t in range(lookback, len(r)):
        y = r[t - lookback:t]
        try:
            # model='avgarch' implements the TARCH/ZARCH logic
            am = arch_model(y, vol='AVGARCH', p=1, o=1, q=1, dist='studentst', rescale=False)
            res = am.fit(disp="off", show_warning=False)
            f_next = res.forecast(horizon=1).variance.iloc[-1, 0]
        except:
            f_next = np.mean(r[t - lookback:t] ** 2)
        f_rollB.append(f_next)
        y_true.append(r[t] ** 2)
        idx.append(t)
    return np.array(f_rollB), np.array(y_true), np.array(idx)



def estimate_midas(Xq_df, return_col="rt", lookback=10):
    f_rollB, y_true, idx = [], [], []
    r = Xq_df[return_col].astype(float).to_numpy()
    
    # Pre-calculate Almon weights (decaying influence)
    # This weights the most recent hours more heavily than the 10th hour
    weights = np.exp(-0.1 * np.arange(lookback))
    weights /= weights.sum()

    for t in range(lookback, len(r)):
        y_window = r[t - lookback:t] ** 2
        
        # MIDAS forecast is the weighted sum of past squared returns
        f_next = np.sum(weights * y_window)

        f_rollB.append(f_next)
        y_true.append(r[t] ** 2)
        idx.append(t)
    return np.array(f_rollB), np.array(y_true), np.array(idx)


import numpy as np
import pandas as pd
from arch.univariate import ARCHInMean, GARCH, Normal

def estimate_garch_m(series):
    model = ARCHInMean(
        y=series,
        volatility=GARCH(p=1, q=1),
        distribution=Normal(),
        form="var",
        rescale=False
    )

    res = model.fit(disp="off", show_warning=False,options={"maxiter": 1000})

    params = res.params

    omega = float(params["omega"])
    alpha_name = [k for k in params.index if k.startswith("alpha[")][0]
    beta_name  = [k for k in params.index if k.startswith("beta[")][0]

    alpha = float(params[alpha_name])
    beta  = float(params[beta_name])

    last_resid = float(res.resid.iloc[-1] if hasattr(res.resid, "iloc") else res.resid[-1])
    last_h = float(
        (res.conditional_volatility.iloc[-1] if hasattr(res.conditional_volatility, "iloc")
         else res.conditional_volatility[-1]) ** 2
    )

    # Manual 1-step-ahead GARCH(1,1) variance forecast
    h_next = omega + alpha * (last_resid ** 2) + beta * last_h
    return h_next

