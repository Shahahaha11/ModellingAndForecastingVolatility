
import numpy as np
from arch import arch_model
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.structural import UnobservedComponents
from arch.univariate import ARCHInMean, GARCH, Normal
import pandas as pd

def _correct_with_exog(
    f_hist,
    y_hist,
    exog_hist,
    f_now,
    exog_now,
    correction_lookback,
    eps=1e-12
):
    if not np.isfinite(f_now) or f_now <= 0 or not np.isfinite(exog_now) or exog_now <= 0:
        return f_now

    y_train = np.asarray(y_hist[-correction_lookback:], dtype=float)
    f_train = np.asarray(f_hist[-correction_lookback:], dtype=float)
    exog_train = np.asarray(exog_hist[-correction_lookback:], dtype=float)

    valid = (
        np.isfinite(y_train)
        & np.isfinite(f_train)
        & np.isfinite(exog_train)
        & (y_train > eps)
        & (f_train > eps)
        & (exog_train > eps)
    )

    if valid.sum() < 3:
        return f_now

    X_train = np.column_stack([
        np.ones(valid.sum()),
        np.log(f_train[valid]),
        np.log(exog_train[valid])
    ])

    try:
        beta = np.linalg.lstsq(
            X_train,
            np.log(y_train[valid]),
            rcond=None
        )[0]
    except np.linalg.LinAlgError:
        return f_now

    if not np.isfinite(beta).all():
        return f_now

    x_now = np.array([
        1.0,
        np.log(max(f_now, eps)),
        np.log(max(exog_now, eps))
    ])

    log_forecast = float(x_now @ beta)
    
    # Catch raw overflows early
    if log_forecast > 50:
        log_forecast = 50
        
    f_corrected = np.exp(log_forecast)

    # --- Dynamic Guardrail Adjustment ---
    # Find the peak tail volatility ratio observed in the current training window
    max_historical_ratio = np.max(y_train[valid] / f_train[valid])
    
    # Allow the model to scale up to 2x the highest recently observed jump, capped at 50x
    dynamic_upper_multiplier = min(max(max_historical_ratio * 2.0, 5.0), 50.0)
    
    lower_bound = f_now * 0.05
    upper_bound = f_now * dynamic_upper_multiplier
    
    return np.clip(f_corrected, lower_bound, upper_bound)

def estimate_naive10_mfiv(
    Xq_df,
    exog_col="MFIV",
    return_col="rt",
    target_col="rv",
    lookback=10,
    correction_lookback=10
):
    f_rollB, y_true, idx = [], [], []
    f_hist, y_hist, exog_hist = [], [], []

    r = Xq_df[return_col].astype(float).to_numpy()
    rv = Xq_df[target_col].astype(float).to_numpy()
    exog = Xq_df[exog_col].astype(float).to_numpy()

    for i in range(lookback, len(r)):
        f_base = r[i - lookback:i].var(ddof=1)

        if len(f_hist) >= correction_lookback:
            f_rollB.append(
                _correct_with_exog(
                    f_hist,
                    y_hist,
                    exog_hist,
                    f_base,
                    exog[i],
                    correction_lookback
                )
            )
            y_true.append(rv[i])
            idx.append(i)

        f_hist.append(f_base)
        y_hist.append(rv[i])
        exog_hist.append(exog[i])

    return np.array(f_rollB), np.array(y_true), np.array(idx)


# def estimate_garch_mfiv(
#     Xq_df,
#     exog_col="MFIV",
#     return_col="rt",
#     target_col="rv",
#     lookback=20,
#     correction_lookback=20
# ):
#     f_rollB, y_true, idx = [], [], []
#     f_hist, y_hist, exog_hist = [], [], []

#     r = Xq_df[return_col].astype(float).to_numpy()
#     rv = Xq_df[target_col].astype(float).to_numpy()
#     exog = Xq_df[exog_col].astype(float).to_numpy()

#     for i in range(lookback, len(r)):
#         r_sub = r[i - lookback:i]

#         am = arch_model(
#             r_sub,
#             mean="Constant",
#             vol="GARCH",
#             p=1,
#             q=1,
#             rescale=False
#         )

#         res = am.fit(disp="off", show_warning=False)
#         f_base = res.forecast(horizon=1).variance.values[-1, 0]

#         if len(f_hist) >= correction_lookback:
#             f_rollB.append(
#                 _correct_with_exog(
#                     f_hist,
#                     y_hist,
#                     exog_hist,
#                     f_base,
#                     exog[i],
#                     correction_lookback
#                 )
#             )
#             y_true.append(rv[i])
#             idx.append(i)

#         f_hist.append(f_base)
#         y_hist.append(rv[i])
#         exog_hist.append(exog[i])

#     return np.array(f_rollB), np.array(y_true), np.array(idx)

######################################################################
# def estimate_garch_mfiv(
#     Xq_df,
#     exog_col="MFIV",
#     return_col="rt",
#     target_col="rv",
#     lookback=20,
#     correction_lookback=20,
#     exog_lag=19  # <--- 1. Add your chosen lag length as an input parameter
# ):
#     f_rollB, y_true, idx = [], [], []
#     f_hist, y_hist, exog_hist = [], [], []

#     r = Xq_df[return_col].astype(float).to_numpy()
#     rv = Xq_df[target_col].astype(float).to_numpy()
#     exog = Xq_df[exog_col].astype(float).to_numpy()

#     # 2. Adjust loop start to ensure we don't look back past index 0 for the lagged exog
#     start_idx = max(lookback, exog_lag)

#     for i in range(start_idx, len(r)):
#         r_sub = r[i - lookback:i]

#         am = arch_model(
#             r_sub,
#             mean="Constant",
#             vol="GARCH",
#             p=1,
#             q=1,
#             rescale=False
#         )

#         res = am.fit(disp="off", show_warning=False)
#         f_base = res.forecast(horizon=1).variance.values[-1, 0]

#         # 3. Pull the exogenous value from the lagged position
#         current_exog = exog[i - exog_lag]

#         if len(f_hist) >= correction_lookback:
#             f_rollB.append(
#                 _correct_with_exog(
#                     f_hist,
#                     y_hist,
#                     exog_hist,
#                     f_base,
#                     current_exog,  # <--- Pass the lagged value
#                     correction_lookback
#                 )
#             )
#             y_true.append(rv[i])
#             idx.append(i)

#         f_hist.append(f_base)
#         y_hist.append(rv[i])
#         exog_hist.append(current_exog)  # <--- Keep the history aligned with the lag

#     return np.array(f_rollB), np.array(y_true), np.array(idx)

################################################
def estimate_garch_mfiv(
    Xq_df,
    exog_col="MFIV",
    return_col="rt",
    target_col="rv",
    lookback=20,
    correction_lookback=20,
    exog_lag=19
):
    f_rollB, y_true, idx = [], [], []
    f_hist, y_hist, exog_hist = [], [], []

    r = Xq_df[return_col].astype(float).to_numpy() * 100.0   # scale to percent-return magnitude
    rv = Xq_df[target_col].astype(float).to_numpy()
    exog = Xq_df[exog_col].astype(float).to_numpy()

    start_idx = max(lookback, exog_lag)

    for i in range(start_idx, len(r)):
        r_sub = r[i - lookback:i]

        am = arch_model(
            r_sub,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1
            # rescale removed: with returns already scaled to ~O(1) magnitude,
            # arch's internal auto-rescale is well-behaved and no longer fights the fit
        )

        res = am.fit(disp="off", show_warning=False)
        f_base = res.forecast(horizon=1).variance.values[-1, 0] / 10000.0  # undo the *100 scaling (100^2)

        current_exog = exog[i - exog_lag]

        if len(f_hist) >= correction_lookback:
            f_rollB.append(
                _correct_with_exog(
                    f_hist,
                    y_hist,
                    exog_hist,
                    f_base,
                    current_exog,
                    correction_lookback
                )
            )
            y_true.append(rv[i])
            idx.append(i)

        f_hist.append(f_base)
        y_hist.append(rv[i])
        exog_hist.append(current_exog)

    return np.array(f_rollB), np.array(y_true), np.array(idx)
    

def estimate_xgboost_par_mfiv(
    Xq_df,
    exog_col="MFIV",
    target_col="rv",
    lookback=20,
    n_lags=5,
    **xgb_kwargs
):
    df = Xq_df[[target_col, exog_col]].copy().dropna()
    df["original_position"] = Xq_df.index.get_indexer(df.index)
    df = df.rename(columns={target_col: "y", exog_col: "_EXOG"}).reset_index(drop=True)

    for lag in range(1, n_lags + 1):
        df[f"lag{lag}"] = df["y"].shift(lag)

    df = df.dropna().reset_index(drop=True)

    feature_cols = [f"lag{lag}" for lag in range(1, n_lags + 1)] + ["_EXOG"]

    params = dict(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=11
    )
    params.update(xgb_kwargs)

    f_rollB, y_true, idx = [], [], []

    for t in range(lookback, len(df)):
        train = df.iloc[t - lookback:t]
        test = df.iloc[[t]]

        model = XGBRegressor(**params)
        model.fit(train[feature_cols], train["y"])

        f_rollB.append(model.predict(test[feature_cols])[0])
        y_true.append(test["y"].iloc[0])
        idx.append(int(test["original_position"].iloc[0]))

    return np.array(f_rollB), np.array(y_true), np.array(idx)


def estimate_sv_mfiv(
    Xq_df,
    exog_col="MFIV",
    return_col="rt",
    target_col="rv",
    lookback=10,
    correction_lookback=10,
    eps=1e-6
):
    f_rollB, y_true, idx = [], [], []
    f_hist, y_hist, exog_hist = [], [], []

    r = Xq_df[return_col].astype(float).to_numpy()
    r_scaled = r * 100.0
    rv = Xq_df[target_col].astype(float).to_numpy()
    exog = Xq_df[exog_col].astype(float).to_numpy()

    for t in range(lookback, len(r)):
        y = np.log(r_scaled[t - lookback:t] ** 2 + eps)

        try:
            res = UnobservedComponents(y, level="local level").fit(disp=False)
            m = float(res.forecast(1)[0])
            v = (
                float(res.filter_results.forecasts_error_cov[0, 0, -1])
                if res.filter_results.forecasts_error_cov.size
                else 0.0
            )
            f_base = np.exp(m + 0.5 * max(v, 0.0)) / 10000.0
        except Exception:
            f_base = float(np.mean(r[t - lookback:t] ** 2))

        if len(f_hist) >= correction_lookback:
            f_rollB.append(
                _correct_with_exog(
                    f_hist,
                    y_hist,
                    exog_hist,
                    f_base,
                    exog[t],
                    correction_lookback
                )
            )
            y_true.append(rv[t])
            idx.append(t)

        f_hist.append(f_base)
        y_hist.append(rv[t])
        exog_hist.append(exog[t])

    return np.array(f_rollB), np.array(y_true), np.array(idx)


def estimate_garch_variant_mfiv(
    Xq_df,
    exog_col="MFIV",
    return_col="rt",
    target_col="rv",
    lookback=10,
    correction_lookback=10,
    variant="GJR"
):
    f_rollB, y_true, idx = [], [], []
    f_hist, y_hist, exog_hist = [], [], []

    r = Xq_df[return_col].astype(float).to_numpy()
    r_scaled = r * 100.0
    rv = Xq_df[target_col].astype(float).to_numpy()
    exog = Xq_df[exog_col].astype(float).to_numpy()

    for t in range(lookback, len(r)):
        window = r_scaled[t - lookback:t]

        try:
            if variant == "GJR":
                model = arch_model(
                    window,
                    p=1,
                    o=1,
                    q=1,
                    vol="GARCH",
                    dist="t"
                )
            elif variant == "EGARCH":
                model = arch_model(
                    window,
                    p=1,
                    o=1,
                    q=1,
                    vol="EGARCH",
                    dist="t"
                )
            else:
                raise ValueError("variant must be 'GJR' or 'EGARCH'")

            res = model.fit(disp="off", show_warning=False)
            f_base = res.forecast(horizon=1).variance.values[-1, 0] / 10000.0
        except Exception:
            f_base = float(np.mean(r[t - lookback:t] ** 2))

        if len(f_hist) >= correction_lookback:
            f_rollB.append(
                _correct_with_exog(
                    f_hist,
                    y_hist,
                    exog_hist,
                    f_base,
                    exog[t],
                    correction_lookback
                )
            )
            y_true.append(rv[t])
            idx.append(t)

        f_hist.append(f_base)
        y_hist.append(rv[t])
        exog_hist.append(exog[t])

    return np.array(f_rollB), np.array(y_true), np.array(idx)


def estimate_gjr_mfiv(Xq_df, exog_col="MFIV", **kwargs):
    return estimate_garch_variant_mfiv(
        Xq_df,
        exog_col=exog_col,
        variant="GJR",
        **kwargs
    )


def estimate_egarch_mfiv(Xq_df, exog_col="MFIV", **kwargs):
    return estimate_garch_variant_mfiv(
        Xq_df,
        exog_col=exog_col,
        variant="EGARCH",
        **kwargs
    )


def estimate_tarch_mfiv(
    Xq_df,
    exog_col="MFIV",
    return_col="rt",
    target_col="rv",
    lookback=10,
    correction_lookback=10
):
    f_rollB, y_true, idx = [], [], []
    f_hist, y_hist, exog_hist = [], [], []

    r = Xq_df[return_col].astype(float).to_numpy()
    rv = Xq_df[target_col].astype(float).to_numpy()
    exog = Xq_df[exog_col].astype(float).to_numpy()

    for t in range(lookback, len(r)):
        window = r[t - lookback:t]

        try:
            am = arch_model(
                window,
                vol="AVGARCH",
                p=1,
                o=1,
                q=1,
                dist="studentst",
                rescale=False
            )
            res = am.fit(disp="off", show_warning=False)
            f_base = res.forecast(horizon=1).variance.iloc[-1, 0]
        except Exception:
            f_base = float(np.mean(window ** 2))

        if len(f_hist) >= correction_lookback:
            f_rollB.append(
                _correct_with_exog(
                    f_hist,
                    y_hist,
                    exog_hist,
                    f_base,
                    exog[t],
                    correction_lookback
                )
            )
            y_true.append(rv[t])
            idx.append(t)

        f_hist.append(f_base)
        y_hist.append(rv[t])
        exog_hist.append(exog[t])

    return np.array(f_rollB), np.array(y_true), np.array(idx)


def estimate_midas_mfiv(
    Xq_df,
    exog_col="MFIV",
    return_col="rt",
    target_col="rv",
    lookback=10,
    correction_lookback=10
):
    f_rollB, y_true, idx = [], [], []
    f_hist, y_hist, exog_hist = [], [], []

    r = Xq_df[return_col].astype(float).to_numpy()
    rv = Xq_df[target_col].astype(float).to_numpy()
    exog = Xq_df[exog_col].astype(float).to_numpy()

    weights = np.exp(-0.1 * np.arange(lookback))
    weights /= weights.sum()

    for t in range(lookback, len(r)):
        f_base = np.sum(weights * r[t - lookback:t] ** 2)

        if len(f_hist) >= correction_lookback:
            f_rollB.append(
                _correct_with_exog(
                    f_hist,
                    y_hist,
                    exog_hist,
                    f_base,
                    exog[t],
                    correction_lookback
                )
            )
            y_true.append(rv[t])
            idx.append(t)

        f_hist.append(f_base)
        y_hist.append(rv[t])
        exog_hist.append(exog[t])

    return np.array(f_rollB), np.array(y_true), np.array(idx)


def estimate_garch_m(series):
    model = ARCHInMean(
        y=series,
        volatility=GARCH(p=1, q=1),
        distribution=Normal(),
        form="var",
        rescale=False
    )

    res = model.fit(
        disp="off",
        show_warning=False,
        options={"maxiter": 1000}
    )

    params = res.params
    omega = float(params["omega"])
    alpha = float(params[[k for k in params.index if k.startswith("alpha[")][0]])
    beta = float(params[[k for k in params.index if k.startswith("beta[")][0]])

    last_resid = float(res.resid.iloc[-1] if hasattr(res.resid, "iloc") else res.resid[-1])
    last_h = float(
        res.conditional_volatility.iloc[-1] ** 2
        if hasattr(res.conditional_volatility, "iloc")
        else res.conditional_volatility[-1] ** 2
    )

    return omega + alpha * last_resid ** 2 + beta * last_h


def estimate_garch_m_mfiv(
    Xq_df,
    exog_col="MFIV",
    return_col="rt",
    target_col="rv",
    min_obs=10,
    correction_lookback=10
):
    f_rollB, y_true, idx = [], [], []
    f_hist, y_hist, exog_hist = [], [], []

    returns = Xq_df[return_col].astype(float) * 100.0
    rv = Xq_df[target_col].astype(float).to_numpy()
    exog = Xq_df[exog_col].astype(float).to_numpy()

    for t in range(min_obs, len(returns)):
        f_base = estimate_garch_m(returns.iloc[:t]) / 10000.0

        if len(f_hist) >= correction_lookback:
            f_rollB.append(
                _correct_with_exog(
                    f_hist,
                    y_hist,
                    exog_hist,
                    f_base,
                    exog[t],
                    correction_lookback
                )
            )
            y_true.append(rv[t])
            idx.append(t)

        f_hist.append(f_base)
        y_hist.append(rv[t])
        exog_hist.append(exog[t])

    return np.array(f_rollB), np.array(y_true), np.array(idx)