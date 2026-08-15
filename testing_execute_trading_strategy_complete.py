### NEW 07
### VI
import pandas as pd
import numpy as np

def testing_execute_trading_strategy(
    bt_year,
    bt_quarter,
    model,
    outputs_by_year,
    market,
    quantity=10,
    lookback_hours=60,
    mean_hours=60,
    hold_days=7,
    expiry_days=30,
    ql=0.1,
    qh=0.9,
    z=1.645
):
    out_full = outputs_by_year[bt_year][bt_quarter].copy()
    out_full.index = pd.to_datetime(out_full.index,utc=True).floor("h")

    out_full = out_full.loc[~out_full.index.duplicated(keep="first")]

    now_list = out_full.index
    snapshots = []
    greek_tests = []

    MFIV_HORIZONS = [191, 382, 573, 764]

    VALID_MODELS = [
        "garch",
        "garch_mfiv",
        "garch_n_mfiv",
        "garch_mfiv_avg",
        "regime_switching_mfiv"
    ]

    if model not in VALID_MODELS:
        raise ValueError(f"model must be one of {VALID_MODELS}")

    rv = out_full["rv"].astype(float)

    use_dynamic_hold = model == "garch_n_mfiv"
    use_dynamic_mfiv_signal = ( model == "regime_switching_mfiv" )

    if model == "garch":
        garch_series = out_full["garch"].astype(float)

    elif model == "garch_mfiv":
        garch_series = out_full["garch_mfiv"].astype(float)

    elif model == "garch_n_mfiv":
        garch_series = out_full["garch"].astype(float)

    elif model == "garch_mfiv_avg":
        garch_series = out_full["garch_mfiv_avg"].astype(float)

    else:
        garch_series = None

    regime_state = pd.to_numeric(out_full["regime_state"], errors="coerce")

    selected_horizon_series = out_full["selected_horizon"].astype("string")

# when do we use this mfiv signal . 
    if use_dynamic_mfiv_signal:
        signal = pd.Series(np.nan, index=out_full.index, dtype=float)

        rolling_forecast_mean = pd.Series(np.nan, index=out_full.index, dtype=float)

        q_lo_series = pd.Series(np.nan, index=out_full.index, dtype=float)

        q_hi_series = pd.Series(np.nan, index=out_full.index, dtype=float)

        forecast_top10_series = pd.Series(np.nan, index=out_full.index, dtype=float)

        forecast_bot10_series = pd.Series(np.nan, index=out_full.index, dtype=float)

        for H in MFIV_HORIZONS:
            col = f"MFIV_{H}"

            mfiv_h = out_full[col].astype(float)

            # Cumulative realised variance known before time t:
            # rv[t-H] + ... + rv[t-1]
            trailing_rv_h = (rv.rolling(window=H, min_periods=H).sum().shift(1))

            signal_h = (mfiv_h - trailing_rv_h) / trailing_rv_h

            forecast_mean_h = mfiv_h.rolling(window=H, min_periods=H).mean()

            q_lo_h = signal_h.rolling( window=H, min_periods=H).quantile(ql)

            q_hi_h = signal_h.rolling(window=H, min_periods=H).quantile(qh) # 90th quantile for instance of the signal

            forecast_top10_h = (forecast_mean_h.rolling(window=H,min_periods=H).quantile(qh))

            forecast_bot10_h = (forecast_mean_h.rolling(window=H, min_periods=H).quantile(ql))

            mask = (selected_horizon_series == col)

            signal.loc[mask] = signal_h.loc[mask]

            rolling_forecast_mean.loc[mask] = forecast_mean_h.loc[mask]

            q_lo_series.loc[mask] = q_lo_h.loc[mask]

            q_hi_series.loc[mask] = q_hi_h.loc[mask]

            forecast_top10_series.loc[mask] = forecast_top10_h.loc[mask]

            forecast_bot10_series.loc[mask] = forecast_bot10_h.loc[mask]

    else:
        signal = (garch_series - rv) / rv

        q_lo_series = signal.rolling(window=lookback_hours, min_periods=lookback_hours).quantile(ql)

        q_hi_series = signal.rolling(window=lookback_hours, min_periods=lookback_hours).quantile(qh)

        rolling_forecast_mean = (garch_series.rolling(window=mean_hours, min_periods=mean_hours).mean())

        forecast_top10_series = (rolling_forecast_mean.rolling(window=lookback_hours, min_periods=lookback_hours).quantile(qh))

        forecast_bot10_series = (rolling_forecast_mean.rolling(window=lookback_hours, min_periods=lookback_hours).quantile(ql))

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
        for instrument in list(
            market.holdings.instruments
        ):
            key = (
                instrument.Strike,
                instrument.Expiry,
                instrument.Right,
                instrument.Direction
            )

            market.liquidate(
                key,
                instrument
            )

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
                    "entry_price": instrument.Price
                }
                for instrument
                in market.holdings.instruments
            ],
            "orders": list(market.orders)
        })

    for idx, now in enumerate(now_list):
        market.update(now)

        if market.get_free_liquidity() < 0:
            liquidate_all()
            reset_position_state()

            market.record_equity()
            save_snapshot(now)
            continue

        if ( not use_dynamic_mfiv_signal and idx < lookback_hours):
            market.record_equity()
            save_snapshot(now)
            continue

        s_now = signal.iloc[idx]
        q_lo = q_lo_series.iloc[idx]
        q_hi = q_hi_series.iloc[idx]

        forecast_mean_now = (rolling_forecast_mean.iloc[idx])

        forecast_top10 = (forecast_top10_series.iloc[idx])

        forecast_bot10 = (forecast_bot10_series.iloc[idx])

        if (
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

        call_count = len(short_calls_held)
        put_count = len(short_puts_held)

        short_position = ( call_count > 0 or put_count > 0)

        long_position = ( len(long_calls_held) > 0 or len(long_puts_held) > 0 )

        has_position = ( short_position or long_position)

        complete_position = (
            (
                call_count > 0
                and put_count > 0
            )
            or (
                len(long_calls_held) > 0
                and len(long_puts_held) > 0
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
        if (
            has_position
            and not complete_position
        ):
            liquidate_all()
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

            if (entered_day is not None and now.date() > entered_day):
                if ( active_position_side == "long_straddle" and forecast_mean_now >= forecast_top10 ):
                    should_exit = True

                elif ( active_position_side == "short_strangle" and forecast_mean_now <= forecast_bot10 ):
                    should_exit = True

            if any( instrument.Expiry - now <= pd.Timedelta(hours=12) for instrument in holdings ):
                should_exit = True

            if should_exit:
                liquidate_all()
                reset_position_state()

                market.record_equity()
                save_snapshot(now)
                continue

        if complete_position:
            market.record_equity()
            save_snapshot(now)
            continue

        # long_entry_signal = ( s_now >= q_hi and 15 <= hour < 16 ) # testing now
        long_entry_signal = ( s_now >= q_hi and 11 <= hour < 15 ) # CHANGED_040826 untested, testing temp first

        short_entry_signal = ( s_now <= q_lo and 11 <= hour < 15 )

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
                )
            ]

            puts = [instrument for instrument in market.chain.instruments
                if (
                    instrument.Right == "P"
                    and instrument.Direction == "buy"
                )
            ]

            valid_calls = [instrument for instrument in calls
                if (
                    pd.Timedelta(days=1) <= instrument.Expiry - now <= pd.Timedelta( days=expiry_days)
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

            before_count = len(market.holdings.instruments)

            market.buy( long_call, qty=1 )

            market.buy( long_put, qty=1 )

            after_count = len( market.holdings.instruments )

            if ( after_count - before_count == 2):
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

            else:
                liquidate_all()
                reset_position_state()

            market.record_equity()
            save_snapshot(now)
            continue


        # ---------------------------------------------------------
        # STRIKE HORIZON
        # ---------------------------------------------------------
        if model in ["garch", "garch_mfiv_avg"]:
            strike_horizon = None
            mfiv_now = None

            lower_boundary = last * 0.98
            upper_boundary = last * 1.02

        else:
            if entry_selected_horizon is not None:
                strike_horizon = entry_selected_horizon

            elif model in ["regime_switching_mfiv", "garch_n_mfiv"]:
                selected_col = selected_horizon_series.iloc[idx]

                if pd.isna(selected_col):
                    market.record_equity()
                    save_snapshot(now)
                    continue

                selected_col = str(selected_col)

                try:
                    strike_horizon = int(selected_col.split("_")[1])
                except (
                    IndexError,
                    TypeError,
                    ValueError
                ):
                    market.record_equity()
                    save_snapshot(now)
                    continue

                if strike_horizon not in MFIV_HORIZONS:
                    market.record_equity()
                    save_snapshot(now)
                    continue

            else:
                strike_horizon = 191

            mfiv_col = f"MFIV_{strike_horizon}"

            if mfiv_col not in out_full.columns:
                market.record_equity()
                save_snapshot(now)
                continue

            mfiv_now = float(out_full[mfiv_col].iloc[idx])

            if not np.isfinite(mfiv_now) or mfiv_now <= 0:
                market.record_equity()
                save_snapshot(now)
                continue

            lower_boundary = (
                last
                * np.exp(
                    -z * np.sqrt(mfiv_now)
                )
            )

            upper_boundary = (
                last
                * np.exp(
                    z * np.sqrt(mfiv_now)
                )
            )

        acceptable_calls = [
            instrument
            for instrument
            in market.chain.instruments
            if (
                instrument.Right == "C"
                and instrument.Direction == "sell"
                and instrument.Strike
                >= upper_boundary
                and pd.Timedelta(days=1)
                <= instrument.Expiry - now
                <= pd.Timedelta(
                    days=expiry_days
                )
            )
        ]

        acceptable_puts = [
            instrument
            for instrument
            in market.chain.instruments
            if (
                instrument.Right == "P"
                and instrument.Direction == "sell"
                and instrument.Strike
                <= lower_boundary
                and pd.Timedelta(days=1)
                <= instrument.Expiry - now
                <= pd.Timedelta(
                    days=expiry_days
                )
            )
        ]
        # ---------------------------------------------------------
        # STRICT TWO-LEG RANGE FILL
        # ---------------------------------------------------------
        selected_calls = []
        selected_puts = []

        common_expiries = sorted(
            {
                instrument.Expiry
                for instrument in acceptable_calls
            }
            & {
                instrument.Expiry
                for instrument in acceptable_puts
            },
            key=lambda expiry_value: abs(
                (expiry_value - now)
                - pd.Timedelta(days=expiry_days)
            )
        )

        # Prefer one common expiry for both legs.
        for expiry in common_expiries:
            expiry_calls = sorted(
                [
                    instrument
                    for instrument in acceptable_calls
                    if instrument.Expiry == expiry
                ],
                key=lambda instrument: instrument.Strike
            )

            expiry_puts = sorted(
                [
                    instrument
                    for instrument in acceptable_puts
                    if instrument.Expiry == expiry
                ],
                key=lambda instrument: instrument.Strike,
                reverse=True
            )

            if expiry_calls and expiry_puts:
                selected_calls = expiry_calls[:quantity]
                selected_puts = expiry_puts[:quantity]
                break

        # Fallback: allow different expiries only when
        # no common-expiry range exists.
        if not selected_calls or not selected_puts:
            if acceptable_calls:
                call_expiry = min(
                    {
                        instrument.Expiry
                        for instrument in acceptable_calls
                    },
                    key=lambda expiry_value: abs(
                        (expiry_value - now)
                        - pd.Timedelta(days=expiry_days)
                    )
                )

                selected_calls = sorted(
                    [
                        instrument
                        for instrument in acceptable_calls
                        if instrument.Expiry == call_expiry
                    ],
                    key=lambda instrument: instrument.Strike
                )[:quantity]

            if acceptable_puts:
                put_expiry = min(
                    {
                        instrument.Expiry
                        for instrument in acceptable_puts
                    },
                    key=lambda expiry_value: abs(
                        (expiry_value - now)
                        - pd.Timedelta(days=expiry_days)
                    )
                )

                selected_puts = sorted(
                    [
                        instrument
                        for instrument in acceptable_puts
                        if instrument.Expiry == put_expiry
                    ],
                    key=lambda instrument: instrument.Strike,
                    reverse=True
                )[:quantity]

        # # Never attempt only one side.
        # if not selected_calls or not selected_puts:
        #     market.record_equity()
        #     save_snapshot(now)
        #     continue 
        
        # CHANGED_050826
        # Enforce the same number of call and put legs.
        pair_count = min(len(selected_calls), len(selected_puts), quantity)

        if pair_count == 0:
            market.record_equity()
            save_snapshot(now)
            continue

        selected_calls = selected_calls[:pair_count]
        selected_puts = selected_puts[:pair_count]
        
        
        before_call_count = call_count
        before_put_count = put_count

        # Fill calls from closest above the upper boundary outward.
        for instrument in selected_calls:
            market.sell(
                instrument,
                qty=1
            )

        # Fill puts from closest below the lower boundary outward.
        for instrument in selected_puts:
            market.sell(
                instrument,
                qty=1
            )

        holdings_after = list(
            market.holdings.instruments
        )

        call_count_after = sum(
            1
            for instrument in holdings_after
            if (
                instrument.Right == "C"
                and instrument.Direction == "sell"
            )
        )

        put_count_after = sum(
            1
            for instrument in holdings_after
            if (
                instrument.Right == "P"
                and instrument.Direction == "sell"
            )
        )

        calls_added = (call_count_after - before_call_count)

        puts_added = (put_count_after - before_put_count)

        # Strict two-leg control:
        # both sides must receive at least one fill.
        if calls_added <= 0 or puts_added <= 0 or calls_added!= puts_added:
            liquidate_all()
            reset_position_state()

            market.record_equity()
            save_snapshot(now)
            continue
        entered_day = now.date()
        position_side = "short_strangle"
        entry_time = now

        if use_dynamic_hold:
            selected_col = (selected_horizon_series.iloc[idx])

            if pd.isna(selected_col):
                liquidate_all()
                reset_position_state()

                market.record_equity()
                save_snapshot(now)
                continue

            selected_col = str(selected_col)

            try:
                selected_H = int(selected_col.split("_")[1])
            except (
                IndexError,
                TypeError,
                ValueError
            ):
                liquidate_all()
                reset_position_state()

                market.record_equity()
                save_snapshot(now)
                continue

            if selected_H not in MFIV_HORIZONS:
                liquidate_all()
                reset_position_state()

                market.record_equity()
                save_snapshot(now)
                continue

            position_hold_hours = selected_H

        else:
            position_hold_hours = (hold_days * 24)

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