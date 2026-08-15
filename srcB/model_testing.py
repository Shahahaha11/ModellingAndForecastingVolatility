import pandas as pd
import numpy as np
def regime_switching_mfiv(
    Xq_df,
    MFIV_COLS=["MFIV_191", "MFIV_382", "MFIV_573", "MFIV_764"],
    # horizons=[191, 382, 573, 764],
    return_col="rt",
    target_col="rv",
    refit_every=1,
    min_improvement=0.02
):
    # MFIV_COLS = [f"MFIV_{x}" for x in horizons ]
    rv = Xq_df[target_col].astype(float)
    mfiv = Xq_df[list(MFIV_COLS)].astype(float)

    horizons = {
        col: int(col.split("_")[1])
        for col in MFIV_COLS
    }

    # For an MFIV forecast made at time t:
    #
    # matched_future_rv[col][t]
    #     = rv[t+1] + ... + rv[t+H]
    #
    # This is used only for historical forecast evaluation.
    matched_future_rv = {}

    for col, H in horizons.items():
        matched_future_rv[col] = (
            rv.shift(-1)
              .rolling(window=H, min_periods=H)
              .sum()
              .shift(-(H - 1))
        )

    f_rollB = []
    y_true = []
    idx = []
    regime_state = []
    selected_horizon = []

    selected_col = None

    # The longest horizon needs:
    # H hours for each completed outcome
    # plus H completed forecast-outcome observations.
    start = 2 * max(horizons.values())

    for i in range(start, len(Xq_df)):

        if selected_col is None or (i - start) % refit_every == 0:
            losses = {}

            for col, H in horizons.items():
                history_end = i - H

                forecast_history = mfiv[col].iloc[:history_end]
                realised_history = matched_future_rv[col].iloc[:history_end]

                valid = (
                    forecast_history.notna()
                    & realised_history.notna()
                    & (forecast_history > 0)
                    & (realised_history > 0)
                )

                forecast_history = forecast_history.loc[valid].iloc[-H:]
                realised_history = realised_history.loc[valid].iloc[-H:]

                if len(forecast_history) < H:
                    losses[col] = np.nan
                    continue

                ratio = realised_history / forecast_history

                # QLIKE divergence.
                qlike = ratio - np.log(ratio) - 1

                losses[col] = float(qlike.mean())

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

                    # Avoid switching because of a negligible
                    # difference in historical loss.
                    if best_loss < current_loss * (1 - min_improvement):
                        selected_col = best_col

        if selected_col is None:
            continue

        forecast = mfiv[selected_col].iloc[i]

        if pd.isna(forecast) or forecast <= 0:
            continue

        matched_outcome = matched_future_rv[selected_col].iloc[i]

        f_rollB.append(float(forecast))

        # Future realised variance is used only as the evaluation target.
        # It is not used when selecting the regime at time i.
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
