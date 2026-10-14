### NEW 8
# Result: TCN attempts to reach 10 matched quantity immediately, then at each of the next 3 hourly observations if the same TCN action remains active, without resetting the original 24-hour holding clock.


import pandas as pd
import numpy as np
from srcB.helpers import *
from scipy.stats import norm

def testing_execute_trading_strategy(
    bt_year, bt_quarter, model, outputs_by_year, market, trader_cache,
    run_by="year",
    quantity=10, lookback_hours=60, mean_hours=60,
    hold_days=7, expiry_days=30, ql=0.1, qh=0.9, z=1.645,
    tcn_score_threshold=0.0, forward_hours=24
    ):


    if run_by == "year":
        trade_full = pd.concat(
            [outputs_by_year[bt_year][q] for q in ["q1", "q2", "q3", "q4"]]
        ).sort_index()

        out_full = trader_cache[bt_year].copy()

    elif run_by == "quarter":
        trade_full = outputs_by_year[bt_year][bt_quarter].copy()
        out_full = trader_cache[(bt_year, bt_quarter)].copy()

    else:
        raise ValueError("run_by must be 'year' or 'quarter'")

    trade_full.index = pd.to_datetime(trade_full.index, utc=True).floor("h")
    trade_full = trade_full.loc[~trade_full.index.duplicated(keep="first")]

    out_full.index = pd.to_datetime(out_full.index, utc=True).floor("h")
    out_full = out_full.loc[~out_full.index.duplicated(keep="first")]

    now_list = trade_full.index

    snapshots = []
    greek_tests = []

    VALID_MODELS = [
        "garch",
        "tcn_mfiv"
    ]

    if model not in VALID_MODELS:
        raise ValueError(f"model must be one of {VALID_MODELS}")


    rv = out_full["rv"].astype(float)
    garch_strike_horizon = hold_days * 24
    garch_historical_vol = np.sqrt(rv.rolling( window=garch_strike_horizon, min_periods=garch_strike_horizon).sum().shift(1))
    
    if model == "garch":
        garch_series = out_full["garch"].astype(float)
        
        signal = (garch_series - rv) / rv
        q_lo_series = signal.rolling(window=lookback_hours, min_periods=lookback_hours).quantile(ql)

        q_hi_series = signal.rolling(window=lookback_hours, min_periods=lookback_hours).quantile(qh)

        rolling_forecast_mean = garch_series.rolling(window=mean_hours, min_periods=mean_hours).mean()

        forecast_top10_series = rolling_forecast_mean.rolling(window=lookback_hours, min_periods=lookback_hours).quantile(qh)

        forecast_bot10_series = rolling_forecast_mean.rolling(window=lookback_hours, min_periods=lookback_hours).quantile(ql)

        tcn_action_series = pd.Series( pd.NA, index=out_full.index, dtype="string")

        tcn_best_score_series = pd.Series(np.nan, index=out_full.index, dtype=float)

    else:

        signal = pd.Series(np.nan, index=out_full.index,dtype=float)

        q_lo_series = pd.Series(np.nan, index=out_full.index, dtype=float)

        q_hi_series = pd.Series(np.nan, index=out_full.index, dtype=float)

        rolling_forecast_mean = pd.Series( np.nan, index=out_full.index, dtype=float)

        forecast_top10_series = pd.Series( np.nan, index=out_full.index, dtype=float)

        forecast_bot10_series = pd.Series(np.nan, index=out_full.index, dtype=float)

        tcn_action_series = out_full["tcn_action"].astype("string")

        tcn_best_score_series = pd.to_numeric( out_full["tcn_best_score"], errors="coerce")
    signal = signal.reindex(now_list)
    q_lo_series = q_lo_series.reindex(now_list)
    q_hi_series = q_hi_series.reindex(now_list)

    rolling_forecast_mean = rolling_forecast_mean.reindex(now_list)
    forecast_top10_series = forecast_top10_series.reindex(now_list)
    forecast_bot10_series = forecast_bot10_series.reindex(now_list)

    tcn_action_series = tcn_action_series.reindex(now_list)
    tcn_best_score_series = tcn_best_score_series.reindex(now_list)
    
    entered_day = None
    position_side = None
    entry_time = None
    position_hold_hours = None

    entry_selected_horizon = None
    entry_mfiv = None
    entry_lower_boundary = None
    entry_upper_boundary = None

    def reset_position_state():
        nonlocal entered_day
        nonlocal position_side
        nonlocal entry_time
        nonlocal position_hold_hours
        nonlocal entry_selected_horizon
        nonlocal entry_mfiv
        nonlocal entry_lower_boundary
        nonlocal entry_upper_boundary

        entered_day = None
        position_side = None
        entry_time = None
        position_hold_hours = None

        entry_selected_horizon = None
        entry_mfiv = None
        entry_lower_boundary = None
        entry_upper_boundary = None

    def liquidate_all():
        for instrument in list(market.holdings.instruments):
            key = (
                instrument.Strike,
                instrument.Expiry,
                instrument.Right,
                instrument.Direction
            )

            market.liquidate(key, instrument)

        return len(market.holdings.instruments) == 0
    def tcn_entry_contracts(now, action, forward_hours=24, delta_target=0.20):
        direction = "buy" if action.startswith("long_") else "sell"
        structure = "straddle" if action.endswith("straddle") else "strangle"

        exit_time = now + pd.Timedelta(hours=forward_hours)

        contracts = market.current[
            (market.current["direction"] == direction)
            & (market.current["amount"] >= 0.1)
        ].drop_duplicates("instrument_name")
        
        expiries = sorted(
            pd.Timestamp(expiry)
            for expiry in contracts["maturity"].dropna().unique()
            if pd.Timestamp(expiry) >= exit_time
        )

        for expiry in expiries:

            exp = contracts[contracts["maturity"] == expiry]

            calls = exp[exp["type"] == "C"]
            puts = exp[exp["type"] == "P"]

            if calls.empty or puts.empty:
                continue

            # ---------------------------------------------------------
            # ATM STRADDLE
            # ---------------------------------------------------------
            if structure == "straddle":

                common_strikes = sorted(
                    set(calls["strike"]) & set(puts["strike"])
                )

                if not common_strikes:
                    continue

                atm_strike = min(
                    common_strikes,
                    key=lambda strike: abs(strike - float(market.spot))
                )

                call_symbol = calls.loc[
                    calls["strike"] == atm_strike,
                    "instrument_name"
                ].iloc[0]

                put_symbol = puts.loc[
                    puts["strike"] == atm_strike,
                    "instrument_name"
                ].iloc[0]

            # ---------------------------------------------------------
            # 20-DELTA STRANGLE
            # ---------------------------------------------------------
            else:

                otm_calls = calls[calls["strike"] > float(market.spot)]
                otm_puts = puts[puts["strike"] < float(market.spot)]

                if otm_calls.empty or otm_puts.empty:
                    continue

                call_choices = []

                for _, row in otm_calls.iterrows():

                    iv = float(row["iv"])

                    if not np.isfinite(iv) or iv <= 0:
                        continue

                    try:
                        d1, _ = get_d1_d2(
                            now,
                            float(market.spot),
                            float(row["strike"]),
                            float(market.time_df.loc[now, "rfr"]),
                            iv,
                            expiry
                        )
                    except (ValueError, ZeroDivisionError):
                        continue

                    delta = get_delta(d1, "C", "buy")

                    call_choices.append(
                        (
                            abs(delta - delta_target),
                            row["instrument_name"]
                        )
                    )

                put_choices = []

                for _, row in otm_puts.iterrows():

                    iv = float(row["iv"])

                    if not np.isfinite(iv) or iv <= 0:
                        continue

                    try:
                        d1, _ = get_d1_d2(
                            now,
                            float(market.spot),
                            float(row["strike"]),
                            float(market.time_df.loc[now, "rfr"]),
                            iv,
                            expiry
                        )
                    except (ValueError, ZeroDivisionError):
                        continue

                    delta = get_delta(d1, "P", "buy")

                    put_choices.append(
                        (
                            abs(abs(delta) - delta_target),
                            row["instrument_name"]
                        )
                    )

                if not call_choices or not put_choices:
                    continue

                call_symbol = min(call_choices, key=lambda x: x[0])[1]
                put_symbol = min(put_choices, key=lambda x: x[0])[1]

            call = next(
                (
                    instrument
                    for instrument in market.chain.instruments
                    if (
                        instrument.Symbol == call_symbol
                        and instrument.Direction == direction
                        and instrument.Amount >= 0.1
                    )
                ),
                None
            )
            put = next(
                (
                    instrument
                    for instrument in market.chain.instruments
                    if (
                        instrument.Symbol == put_symbol
                        and instrument.Direction == direction
                        and instrument.Amount >= 0.1
                    )
                ),
                None
            )
            if call is not None and put is not None:
                return call, put

        return None

    def save_snapshot(now):
        snapshots.append({
            "time": now,
            "model": model,
            "cash": market.cash,
            "equity": (market.cash + market.option_value()),
            "option_value": market.option_value(),
            "entry_selected_horizon": (entry_selected_horizon),
            "entry_mfiv": entry_mfiv,
            "entry_lower_boundary": (entry_lower_boundary),
            "entry_upper_boundary": (entry_upper_boundary),
            "positions": [
                {
                    "symbol": instrument.Symbol,
                    "strike": instrument.Strike,
                    "expiry": instrument.Expiry,
                    "right": instrument.Right,
                    "direction": instrument.Direction,
                    "entry_price": instrument.Price,
                    "amount": instrument.Amount,
                }
                for instrument
                in market.holdings.instruments
            ],
            "orders": list(market.orders)
        })

    for idx, now in enumerate(now_list):
        market.update(now)

        # ---------------------------------------------------------
        # YEAR-END LIQUIDATION WINDOW
        # ---------------------------------------------------------
        if now >= pd.Timestamp(f"{bt_year + 1}-01-01", tz="UTC") - pd.Timedelta(days=7):
            if liquidate_all():
                reset_position_state()

            market.record_equity()
            save_snapshot(now)
            continue

        if market.get_free_liquidity() < 0:
            if liquidate_all():
                reset_position_state()

            market.record_equity()
            save_snapshot(now)
            continue

        s_now = signal.iloc[idx]
        q_lo = q_lo_series.iloc[idx]
        q_hi = q_hi_series.iloc[idx]

        forecast_mean_now = (rolling_forecast_mean.iloc[idx])

        forecast_top10 = (forecast_top10_series.iloc[idx])

        forecast_bot10 = (forecast_bot10_series.iloc[idx])

        if model != "tcn_mfiv" and (
            pd.isna(s_now)
            or pd.isna(q_lo)
            or pd.isna(q_hi)
            or pd.isna(forecast_mean_now)
            or pd.isna(forecast_top10)
            or pd.isna(forecast_bot10)
        ): 
            market.record_equity()
            save_snapshot(now)
            continue

        hour = now.hour

        holdings = list(market.holdings.instruments)

        short_calls_held = [instrument for instrument in holdings if ( instrument.Right == "C" and instrument.Direction == "sell" )]

        short_puts_held = [instrument for instrument in holdings if ( instrument.Right == "P" and instrument.Direction == "sell" )]

        long_calls_held = [instrument for instrument in holdings if ( instrument.Right == "C" and instrument.Direction == "buy")]

        long_puts_held = [instrument for instrument in holdings if ( instrument.Right == "P" and instrument.Direction == "buy" )]

        call_count = sum(i.Amount for i in short_calls_held)
        put_count = sum(i.Amount for i in short_puts_held)

        long_call_count = sum(i.Amount for i in long_calls_held)
        long_put_count = sum(i.Amount for i in long_puts_held)

        short_position = (
            call_count > 0
            or put_count > 0
        )

        long_position = (
            long_call_count > 0
            or long_put_count > 0
        )

        has_position = (
            short_position
            or long_position
        )

        if model == "tcn_mfiv" and not has_position and entry_time is not None:
            reset_position_state()

        complete_position = (
            (
                call_count > 0
                and put_count > 0
                and np.isclose(call_count, put_count)
            )
            or (
                long_call_count > 0
                and long_put_count > 0
                and np.isclose(long_call_count, long_put_count)
            )
        )

        active_position_side = position_side
        
        if active_position_side is None:
            if short_position and not long_position:
                active_position_side = "short_strangle"
            elif long_position and not short_position:
                active_position_side = "long_straddle"

        # ---------------------------------------------------------
        # INCOMPLETE POSITION EDGE CASE
        # ---------------------------------------------------------
        if (has_position and not complete_position):
            if liquidate_all():
                reset_position_state()
            market.record_equity()
            save_snapshot(now)
            continue
        # ---------------------------------------------------------
        # EXIT
        # ---------------------------------------------------------
        if has_position:

            should_exit = False

            if (
                entry_time is not None
                and position_hold_hours is not None
                and now >= entry_time + pd.Timedelta(hours=position_hold_hours)
            ):
                should_exit = True

            if (
                model == "garch"
                and entered_day is not None
                and now.date() > entered_day
            ):

                if (
                    active_position_side == "long_straddle"
                    and forecast_mean_now >= forecast_top10
                ):
                    should_exit = True

                elif (
                    active_position_side == "short_strangle"
                    and forecast_mean_now <= forecast_bot10
                ):
                    should_exit = True

            if (
                model == "garch"
                and any(
                    instrument.Expiry - now <= pd.Timedelta(hours=12)
                    for instrument in holdings
                )
            ):
                should_exit = True

            if should_exit:

                if liquidate_all():
                    reset_position_state()

                market.record_equity()
                save_snapshot(now)
                continue

        if complete_position:

            if model == "tcn_mfiv":
                tcn_position_qty = (call_count if short_position else long_call_count)

                if (tcn_position_qty >= float(quantity)
                    or entry_time is None
                    or now > entry_time + pd.Timedelta(hours=3)):
                    market.record_equity()
                    save_snapshot(now)
                    continue

            elif model == "garch":
                garch_position_qty = (call_count if short_position else long_call_count)

                if (garch_position_qty >= float(quantity)
                    or entry_time is None
                    or now > entry_time + pd.Timedelta(hours=3)):
                    market.record_equity()
                    save_snapshot(now)
                    continue
            
        # ---------------------------------------------------------
        # TCN ENTRY
        # ---------------------------------------------------------
        if model == "tcn_mfiv":
            action_now = tcn_action_series.iloc[idx]
            score_now = tcn_best_score_series.iloc[idx]

            if (
                pd.isna(action_now)
                or action_now == "no_trade"
                or pd.isna(score_now)
                or score_now < tcn_score_threshold
            ):
                market.record_equity()
                save_snapshot(now)
                continue

            action_now = str(action_now)
            valid_actions = {
                "long_straddle",
                "short_straddle",
                "long_strangle",
                "short_strangle"
            }

            if action_now not in valid_actions:
                market.record_equity()
                save_snapshot(now)
                continue
            if has_position and action_now != position_side:
                market.record_equity()
                save_snapshot(now)
                continue

            selected = tcn_entry_contracts(
                now,
                action=action_now,
                forward_hours=forward_hours,
                delta_target=0.20
            )

            if selected is None:
                market.record_equity()
                save_snapshot(now)
                continue
            # ADD STARTS
            selected_call, selected_put = selected

            existing_qty = (
                call_count if action_now.startswith("short_")
                else long_call_count
            )

            quantity_left = float(quantity) - existing_qty

            pair_qty = min(
                selected_call.Amount,
                selected_put.Amount,
                quantity_left
            )

            if pair_qty < 0.1:
                market.record_equity()
                save_snapshot(now)
                continue

            call_exists, _, call_cash_change = market.mock_trade(
                selected_call, qty=pair_qty
            )

            put_exists, _, put_cash_change = market.mock_trade(
                selected_put, qty=pair_qty
            )

            if action_now.startswith("long_"):

                trade_exists = (
                    call_exists
                    and put_exists
                    and market.get_free_liquidity()
                    + call_cash_change
                    + put_cash_change >= 0
                )

            else:
                new_margin = (
                    market.spot
                    * (market.im_multiplier + market.mm_buffer_rate)
                    * pair_qty
                    * 2
                )

                trade_exists = (
                    call_exists
                    and market.get_free_liquidity()
                    + call_cash_change
                    + put_cash_change
                    - new_margin >= 0
                )

            if not trade_exists:
                market.record_equity()
                save_snapshot(now)
                continue

            if action_now.startswith("long_"):
                market.buy(selected_call, qty=pair_qty)
                market.buy(selected_put, qty=pair_qty)
            else:
                market.sell(selected_call, qty=pair_qty)
                market.sell(selected_put, qty=pair_qty)
            # END
            if entry_time is None:

                entered_day = now.date()
                position_side = action_now
                entry_time = now
                position_hold_hours = forward_hours
                entry_selected_horizon = forward_hours
                entry_lower_boundary = None
                entry_upper_boundary = None

                if "MFIV_24" in out_full.columns:
                    mfiv_value = float(out_full.loc[now, "MFIV_24"])
                    entry_mfiv = mfiv_value if np.isfinite(mfiv_value) else None
                else:
                    entry_mfiv = None

            call_iv = float(
                market.current.loc[
                    (market.current["instrument_name"] == selected_call.Symbol)
                    & (market.current["direction"] == selected_call.Direction),
                    "iv"
                ].iloc[0]
            )
            put_iv = float(
                market.current.loc[
                    (market.current["instrument_name"] == selected_put.Symbol)
                    & (market.current["direction"] == selected_put.Direction),
                    "iv"
                ].iloc[0]
            )
            rfr_now = float(market.time_df.loc[now, "rfr"])

            greek_tests.append({
                "time": now,
                "tcn_action": action_now,
                "tcn_best_score": float(tcn_best_score_series.iloc[idx]),
                "signal": np.nan,
                "q_lo": np.nan,
                "selected_calls": [selected_call],
                "selected_puts": [selected_put],
                "call_ivs": [call_iv],
                "put_ivs": [put_iv],
                "spot": float(market.spot),
                "rfr": rfr_now
            })
            if np.isfinite(call_iv) and call_iv > 0:

                call_d1, _ = get_d1_d2(
                    now, market.spot, selected_call.Strike,
                    rfr_now, call_iv, selected_call.Expiry
                )

                greek_tests[-1]["call_greeks"] = [{
                    "delta": get_delta(
                        call_d1, selected_call.Right, selected_call.Direction
                    ),
                    "gamma": get_gamma(
                        call_d1, selected_call.Right, now, market.spot,
                        call_iv, selected_call.Expiry, selected_call.Direction
                    )
                }]

            else:
                greek_tests[-1]["call_greeks"] = [{
                    "delta": np.nan,
                    "gamma": np.nan
                }]


            if np.isfinite(put_iv) and put_iv > 0:

                put_d1, _ = get_d1_d2(
                    now, market.spot, selected_put.Strike,
                    rfr_now, put_iv, selected_put.Expiry
                )

                greek_tests[-1]["put_greeks"] = [{
                    "delta": get_delta(
                        put_d1, selected_put.Right, selected_put.Direction
                    ),
                    "gamma": get_gamma(
                        put_d1, selected_put.Right, now, market.spot,
                        put_iv, selected_put.Expiry, selected_put.Direction
                    )
                }]

            else:
                greek_tests[-1]["put_greeks"] = [{
                    "delta": np.nan,
                    "gamma": np.nan
                }]
            market.record_equity()
            save_snapshot(now)
            continue

        long_entry_signal = ( s_now >= q_hi and 11 <= hour < 15 )
        short_entry_signal = ( s_now <= q_lo and 11 <= hour < 15 )

        if has_position and (
            (active_position_side == "long_straddle" and not long_entry_signal)
            or
            (active_position_side == "short_strangle" and not short_entry_signal)
        ):
            market.record_equity()
            save_snapshot(now)
            continue

        if not ( long_entry_signal or short_entry_signal):
            market.record_equity()
            save_snapshot(now)
            continue
        
        last_five = market.close_price(n=5)

        if last_five.empty:
            market.record_equity()
            save_snapshot(now)
            continue

        last = float( last_five.iloc[-1] )

        # ---------------------------------------------------------
        # LONG STRADDLE ENTRY
        # ---------------------------------------------------------
        if long_entry_signal:
            calls = [instrument for instrument in market.chain.instruments
                if (
                    instrument.Right == "C"
                    and instrument.Direction == "buy"
                    and instrument.Amount >= 0.1
                )
            ]

            puts = [instrument for instrument in market.chain.instruments
                if (
                    instrument.Right == "P"
                    and instrument.Direction == "buy"
                    and instrument.Amount >= 0.1
                )
            ]

            valid_calls = [instrument for instrument in calls
                if (
                    pd.Timedelta(days=1) <= (instrument.Expiry - now) <= pd.Timedelta( days=expiry_days)
                )
            ]

            valid_puts = [instrument for instrument in puts
                if (
                    pd.Timedelta(days=1) <= instrument.Expiry - now <= pd.Timedelta(days=expiry_days)
                )
            ]

            if not valid_calls or not valid_puts:
                market.record_equity()
                save_snapshot(now)
                continue

            common_expiries = sorted(
                {
                    instrument.Expiry
                    for instrument in valid_calls
                }
                & {
                    instrument.Expiry
                    for instrument in valid_puts
                }
            )

            if not common_expiries:
                market.record_equity()
                save_snapshot(now)
                continue

            expiry = common_expiries[0]

            expiry_calls = [instrument for instrument in valid_calls if instrument.Expiry == expiry ]

            expiry_puts = [ instrument for instrument in valid_puts if instrument.Expiry == expiry ]

            common_strikes = sorted(
                {
                    instrument.Strike
                    for instrument in expiry_calls
                }
                & {
                    instrument.Strike
                    for instrument in expiry_puts
                }
            )

            if not common_strikes:
                market.record_equity()
                save_snapshot(now)
                continue

            atm_strike = min( common_strikes, key=lambda strike: abs(strike - last) )

            long_call = next(
                (
                    instrument for instrument in expiry_calls if instrument.Strike == atm_strike
                ),
                None
            )

            long_put = next(
                (
                    instrument for instrument in expiry_puts if instrument.Strike == atm_strike
                ),
                None
            )

            if ( long_call is None or long_put is None):
                market.record_equity()
                save_snapshot(now)
                continue

            quantity_left = float(quantity) - long_call_count

            pair_qty = min(
                long_call.Amount,
                long_put.Amount,
                quantity_left
            )

            if pair_qty < 0.1:
                market.record_equity()
                save_snapshot(now)
                continue

            call_exists, _, call_cash_change = market.mock_trade(
                long_call, qty=pair_qty
            )

            put_exists, _, put_cash_change = market.mock_trade(
                long_put, qty=pair_qty
            )

            trade_exists = (
                call_exists
                and put_exists
                and market.get_free_liquidity()
                + call_cash_change
                + put_cash_change >= 0
            )

            if not trade_exists:
                market.record_equity()
                save_snapshot(now)
                continue

            market.buy( long_call, qty=pair_qty )
            market.buy( long_put, qty=pair_qty )

            if entry_time is None:
                entered_day = now.date()
                position_side = ("long_straddle")
                entry_time = now
                position_hold_hours = (hold_days * 24)

            greek_tests.append({
                "time": now,
                "signal": s_now,
                "q_lo": q_lo,
                "selected_calls": [long_call],
                "selected_puts": [long_put],
                "call_ivs": [
                    float(market.current.loc[
                        (market.current["instrument_name"] == long_call.Symbol)
                        & (market.current["direction"] == long_call.Direction),
                        "iv"
                    ].iloc[0])
                ],
                "put_ivs": [
                    float(market.current.loc[
                        (market.current["instrument_name"] == long_put.Symbol)
                        & (market.current["direction"] == long_put.Direction),
                        "iv"
                    ].iloc[0])
                ],
                "spot": float(market.spot),
                "rfr": float(market.time_df.loc[now, "rfr"])
            })

            d1, d2 = get_d1_d2(
                now, market.spot, long_call.Strike,
                greek_tests[-1]["rfr"], greek_tests[-1]["call_ivs"][0], long_call.Expiry
            )

            greek_tests[-1]["call_greeks"] = [{
                "delta": get_delta(d1, long_call.Right, long_call.Direction),
                "gamma": get_gamma(
                    d1, long_call.Right, now, market.spot,
                    greek_tests[-1]["call_ivs"][0], long_call.Expiry, long_call.Direction
                )
            }]

            d1, d2 = get_d1_d2(
                now, market.spot, long_put.Strike,
                greek_tests[-1]["rfr"], greek_tests[-1]["put_ivs"][0], long_put.Expiry
            )

            greek_tests[-1]["put_greeks"] = [{
                "delta": get_delta(d1, long_put.Right, long_put.Direction),
                "gamma": get_gamma(
                    d1, long_put.Right, now, market.spot,
                    greek_tests[-1]["put_ivs"][0], long_put.Expiry, long_put.Direction
                )
            }]

            market.record_equity()
            save_snapshot(now)
            continue
        # ---------------------------------------------------------
        # GARCH STRIKE BOUNDARIES
        # ---------------------------------------------------------


        historical_vol_now = float(garch_historical_vol.loc[now])

        garch_now = float(garch_series.loc[now])

        if (
            not np.isfinite(historical_vol_now)
            or not np.isfinite(garch_now)
            or garch_now <= 0
        ):
            market.record_equity()
            save_snapshot(now)
            continue


        garch_forecast_vol = np.sqrt(garch_now * garch_strike_horizon)

        spot_now = float(market.spot)

        S_minus = spot_now * np.exp(-z * historical_vol_now)

        S_plus = spot_now * np.exp(z * historical_vol_now)

        lower_boundary = S_minus * np.exp(-z * garch_forecast_vol)

        upper_boundary = S_plus * np.exp(z * garch_forecast_vol)
        acceptable_calls = [
            instrument
            for instrument
            in market.chain.instruments
            if (
                instrument.Right == "C"
                and instrument.Direction == "sell"
                and instrument.Strike >= upper_boundary
                and pd.Timedelta(days=1) <= instrument.Expiry - now <= pd.Timedelta(days=expiry_days)
                and instrument.Amount >= 0.1
            )
        ]

        acceptable_puts = [
            instrument
            for instrument
            in market.chain.instruments
            if (
                instrument.Right == "P"
                and instrument.Direction == "sell"
                and instrument.Strike <= lower_boundary
                and pd.Timedelta(days=1) <= instrument.Expiry - now <= pd.Timedelta(days=expiry_days)
                and instrument.Amount >= 0.1
            )
        ]

        # ---------------------------------------------------------
        # STRICT TWO-LEG RANGE FILL
        # ---------------------------------------------------------
        selected_calls = []
        selected_puts = []

        common_expiries = sorted(
            {
                instrument.Expiry for instrument in acceptable_calls
            }
            & {
                instrument.Expiry for instrument in acceptable_puts
            },
            key=lambda expiry_value: abs((expiry_value - now) - pd.Timedelta(days=expiry_days))
        )

        for expiry in common_expiries:
            expiry_calls = sorted(
                [
                    instrument for instrument in acceptable_calls if instrument.Expiry == expiry
                ],
                key=lambda instrument: instrument.Strike, 
                reverse = False
            )

            expiry_puts = sorted(
                [
                    instrument for instrument in acceptable_puts if instrument.Expiry == expiry
                ],
                key=lambda instrument: instrument.Strike,
                reverse=True
            )

            if expiry_calls and expiry_puts:
                selected_calls = expiry_calls
                selected_puts = expiry_puts
                break

        if not selected_calls or not selected_puts:
            if acceptable_calls:
                call_expiry = min(
                    {
                        instrument.Expiry
                        for instrument in acceptable_calls
                    },
                    key=lambda expiry_value: abs( (expiry_value - now) - pd.Timedelta(days=expiry_days))
                )

                selected_calls = sorted(
                    [
                        instrument
                        for instrument in acceptable_calls
                        if instrument.Expiry == call_expiry
                    ],
                    key=lambda instrument: instrument.Strike
                )

            if acceptable_puts:
                put_expiry = min(
                    {
                        instrument.Expiry for instrument in acceptable_puts
                    },
                    key=lambda expiry_value: abs((expiry_value - now) - pd.Timedelta(days=expiry_days))
                )

                selected_puts = sorted(
                    [
                        instrument for instrument in acceptable_puts if instrument.Expiry == put_expiry
                    ],
                    key=lambda instrument: instrument.Strike,
                    reverse=True
                )

        if not selected_calls or not selected_puts:
            market.record_equity()
            save_snapshot(now)
            continue

        selected_call = selected_calls[0]
        selected_put = selected_puts[0]

        quantity_left = float(quantity) - call_count

        pair_qty = min(
            selected_call.Amount,
            selected_put.Amount,
            quantity_left
        )

        if pair_qty < 0.1:
            market.record_equity()
            save_snapshot(now)
            continue

        selected_calls = [selected_call]
        selected_puts = [selected_put]

        call_exists, _, call_cash_change = market.mock_trade(
            selected_call, qty=pair_qty
        )

        _, _, put_cash_change = market.mock_trade(
            selected_put, qty=pair_qty
        )

        new_margin = (
            market.spot
            * (market.im_multiplier + market.mm_buffer_rate)
            * pair_qty
            * 2
        )

        trade_exists = (
            call_exists
            and market.get_free_liquidity()
            + call_cash_change
            + put_cash_change
            - new_margin >= 0
        )

        if not trade_exists:
            market.record_equity()
            save_snapshot(now)
            continue

        market.sell(selected_call, qty=pair_qty)
        market.sell(selected_put, qty=pair_qty)
        if entry_time is None:
            entered_day = now.date()
            position_side = "short_strangle"
            entry_time = now

            position_hold_hours = hold_days * 24

            entry_selected_horizon = (strike_horizon)
            entry_mfiv = mfiv_now
            entry_lower_boundary = (lower_boundary)
            entry_upper_boundary = (upper_boundary)

        greek_tests.append({
            "time": now,
            "signal": s_now,
            "q_lo": q_lo,
            "selected_calls": selected_calls,
            "selected_puts": selected_puts,
            "call_ivs": [
                float(market.current.loc[
                    (market.current["instrument_name"] == instrument.Symbol)
                    & (market.current["direction"] == instrument.Direction),
                    "iv"
                ].iloc[0])
                for instrument in selected_calls
            ],
            "put_ivs": [
                float(market.current.loc[
                    (market.current["instrument_name"] == instrument.Symbol)
                    & (market.current["direction"] == instrument.Direction),
                    "iv"
                ].iloc[0])
                for instrument in selected_puts
            ],
            "spot": float(market.spot),
            "rfr": float(market.time_df.loc[now, "rfr"])
        })

        greek_tests[-1]["call_greeks"] = []
        for instrument, iv in zip(selected_calls, greek_tests[-1]["call_ivs"]):
            d1, d2 = get_d1_d2(
                now, market.spot, instrument.Strike,
                greek_tests[-1]["rfr"], iv, instrument.Expiry
            )
            greek_tests[-1]["call_greeks"].append({
                "delta": get_delta(d1, instrument.Right, instrument.Direction),
                "gamma": get_gamma(
                    d1, instrument.Right, now, market.spot,
                    iv, instrument.Expiry, instrument.Direction
                )
            })

        greek_tests[-1]["put_greeks"] = []
        for instrument, iv in zip(selected_puts, greek_tests[-1]["put_ivs"]):
            d1, d2 = get_d1_d2(
                now, market.spot, instrument.Strike,
                greek_tests[-1]["rfr"], iv, instrument.Expiry
            )
            greek_tests[-1]["put_greeks"].append({
                "delta": get_delta(d1, instrument.Right, instrument.Direction),
                "gamma": get_gamma(
                    d1, instrument.Right, now, market.spot,
                    iv, instrument.Expiry, instrument.Direction
                )
            })

        market.record_equity()
        save_snapshot(now)

    liquidate_all()
    market.record_equity()
    save_snapshot(market.now)

    for test in greek_tests:
        test["entry_equity"] = next(
            s["equity"] for s in snapshots
            if s["time"] == test["time"]
        )

    return snapshots, greek_tests