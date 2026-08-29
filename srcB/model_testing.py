import pandas as pd
import numpy as np

def regime_switching_mfiv(
    Xq_df,
    MFIV_COLS,
    return_col="rt",
    target_col="rv",
    refit_every=1,
    min_improvement=0.02,
    score_window=168
):
    rv = Xq_df[target_col].astype(float)
    mfiv = Xq_df[list(MFIV_COLS)].astype(float)

    horizons = {
        col: int(col.split("_")[1])
        for col in MFIV_COLS
    }

    matched_future_rv = {}
    trailing_rv = {}

    for col, H in horizons.items():

        # Realised variance after forecast time t:
        # rv[t+1] + ... + rv[t+H]
        matched_future_rv[col] = (
            rv.shift(-1)
              .rolling(window=H, min_periods=H)
              .sum()
              .shift(-(H - 1))
        )

        # Naive same-horizon benchmark known at time t:
        # rv[t-H] + ... + rv[t-1]
        trailing_rv[col] = (
            rv.rolling(window=H, min_periods=H)
              .sum()
              .shift(1)
        )

    f_rollB = []
    y_true = []
    idx = []
    regime_state = []
    selected_horizon = []

    selected_col = None

    # Need enough history for:
    # trailing H-hour RV,
    # completed future H-hour RV,
    # and score_window completed comparisons.
    start = 2 * max(horizons.values()) + score_window

    for i in range(start, len(Xq_df)):

        if selected_col is None or (i - start) % refit_every == 0:

            losses = {}

            for col, H in horizons.items():

                # Only forecasts whose full H-hour outcome
                # is known before current time i.
                history_end = i - H

                mfiv_history = mfiv[col].iloc[:history_end]
                naive_history = trailing_rv[col].iloc[:history_end]
                realised_history = matched_future_rv[col].iloc[:history_end]

                valid = (
                    mfiv_history.notna()
                    & naive_history.notna()
                    & realised_history.notna()
                    & (mfiv_history > 0)
                    & (naive_history > 0)
                    & (realised_history > 0)
                )

                mfiv_history = mfiv_history.loc[valid].iloc[-score_window:]
                naive_history = naive_history.loc[valid].iloc[-score_window:]
                realised_history = realised_history.loc[valid].iloc[-score_window:]

                if len(mfiv_history) < score_window:
                    losses[col] = np.nan
                    continue

                mfiv_ratio = realised_history / mfiv_history
                naive_ratio = realised_history / naive_history

                mfiv_qlike = (
                    mfiv_ratio
                    - np.log(mfiv_ratio)
                    - 1
                )

                naive_qlike = (
                    naive_ratio
                    - np.log(naive_ratio)
                    - 1
                )

                # Negative means MFIV beat trailing RV
                # at this same horizon.
                losses[col] = float(
                    (mfiv_qlike - naive_qlike).mean()
                )

            losses = pd.Series(losses, dtype=float).dropna()

            if not losses.empty:

                best_col = losses.idxmin()
                best_loss = losses.loc[best_col]

                if selected_col is None:
                    selected_col = best_col

                elif selected_col not in losses.index:
                    selected_col = best_col

                else:
                    current_loss = losses.loc[selected_col]

                    if best_loss < current_loss - min_improvement:
                        selected_col = best_col

        if selected_col is None:
            continue

        forecast = mfiv[selected_col].iloc[i]

        if pd.isna(forecast) or forecast <= 0:
            continue

        matched_outcome = matched_future_rv[selected_col].iloc[i]

        f_rollB.append(float(forecast))

        y_true.append(
            float(matched_outcome)
            if pd.notna(matched_outcome)
            else np.nan
        )

        idx.append(i)
        regime_state.append(MFIV_COLS.index(selected_col))
        selected_horizon.append(selected_col)

    return (
        np.asarray(f_rollB, dtype=float),
        np.asarray(y_true, dtype=float),
        np.asarray(idx, dtype=int),
        np.asarray(regime_state, dtype=int),
        np.asarray(selected_horizon, dtype=object)
    )