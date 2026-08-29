### NEW 07
### VII
import pandas as pd
import numpy as np
from srcB.helpers import *
from scipy.stats import norm

def testing_execute_trading_strategy(
    bt_year, bt_quarter, model, outputs_by_year, market, trader_cache,
    run_by="year",
    quantity=10, lookback_hours=60, mean_hours=60,
    hold_days=7, expiry_days=30, ql=0.1, qh=0.9, z=1.645,
    risk_fraction=0.01
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

    # MFIV_HORIZONS = [191, 382, 573, 764]
    # MFIV_HORIZONS = list(range(2, 577))
    MFIV_HORIZONS = [12, 24, 72, 168, 336]

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
    garch_strike_horizon = hold_days * 24

    garch_historical_vol = np.sqrt(
        rv.rolling(
            window=garch_strike_horizon,
            min_periods=garch_strike_horizon
        ).sum().shift(1)
    )
    historical_vol = {}

    for H in MFIV_HORIZONS:
        historical_vol[H] = np.sqrt(
            rv.rolling(window=H, min_periods=H).sum().shift(1)
        )
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
            q_hi_h = signal_h.rolling(window=H, min_periods=H).quantile(qh)
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
    
    signal = signal.reindex(now_list)
    q_lo_series = q_lo_series.reindex(now_list)
    q_hi_series = q_hi_series.reindex(now_list)
    rolling_forecast_mean = rolling_forecast_mean.reindex(now_list)
    forecast_top10_series = forecast_top10_series.reindex(now_list)
    forecast_bot10_series = forecast_bot10_series.reindex(now_list)
    selected_horizon_series = selected_horizon_series.reindex(now_list)
    regime_state = regime_state.reindex(now_list)
    
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

    def selected_horizon_now(idx):
        selected_col = selected_horizon_series.iloc[idx]

        if pd.isna(selected_col):
            return None

        try:
            selected_H = int(str(selected_col).split("_")[1])
        except (IndexError, TypeError, ValueError):
            return None

        if selected_H not in MFIV_HORIZONS:
            return None

        return selected_H

    def option_greeks(now, instrument):
        iv_row = market.current.loc[
            (market.current["instrument_name"] == instrument.Symbol)
            & (market.current["direction"] == instrument.Direction),
            "iv"
        ]

        if iv_row.empty:
            return None

        iv = float(iv_row.iloc[0])

        if not np.isfinite(iv) or iv <= 0:
            return None

        d1, _ = get_d1_d2(
            now, market.spot, instrument.Strike,
            float(market.time_df.loc[now, "rfr"]), iv, instrument.Expiry
        )

        return {
            "iv": iv,
            "delta": get_delta(d1, instrument.Right, instrument.Direction),
            "gamma": get_gamma(
                d1, instrument.Right, now, market.spot,
                iv, instrument.Expiry, instrument.Direction
            )
        }

    def entry_fee(instrument):
        return min(0.0003, 0.125 * instrument.OptionPriceBTC) * market.spot

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

        if model in ["garch", "garch_n_mfiv"]:
            missing_signal = (
                pd.isna(s_now)
                or pd.isna(q_lo)
                or pd.isna(q_hi)
            )
        else:
            missing_signal = (
                pd.isna(s_now)
                or pd.isna(q_lo)
                or pd.isna(q_hi)
                or pd.isna(forecast_mean_now)
                or pd.isna(forecast_top10)
                or pd.isna(forecast_bot10)
            )

        if missing_signal:
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
                and call_count == put_count
            )
            or (
                len(long_calls_held) > 0
                and len(long_puts_held) > 0
                and len(long_calls_held) == len(long_puts_held)
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

            if (entry_time is not None
                and position_hold_hours is not None
                and now >= entry_time + pd.Timedelta(hours=position_hold_hours)
            ):
                should_exit = True

            if model in ["garch", "garch_n_mfiv"]:
                # q_hi/q_lo define the opposite variance regime; neutral observations do not force churn.
                if (
                    active_position_side == "long_straddle"
                    and s_now <= q_lo
                ):
                    should_exit = True

                elif (
                    active_position_side == "short_strangle"
                    and s_now >= q_hi
                ):
                    should_exit = True

            elif (entered_day is not None and now.date() > entered_day):
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
                active_position_side == "short_strangle"
                and model in ["garch", "garch_n_mfiv"]
                and entry_lower_boundary is not None
                and entry_upper_boundary is not None
                and (
                    market.spot <= entry_lower_boundary
                    or market.spot >= entry_upper_boundary
                )
            ):
                should_exit = True

            if any( instrument.Expiry - now <= pd.Timedelta(hours=12) for instrument in holdings ):
                should_exit = True

            if should_exit:
                if liquidate_all():
                    reset_position_state()

                market.record_equity()
                save_snapshot(now)
                continue

        if complete_position:
            market.record_equity()
            save_snapshot(now)
            continue

        long_entry_signal = ( s_now >= q_hi and 11 <= hour < 15 )
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
            long_horizon = None

            if model == "garch":
                long_horizon = garch_strike_horizon

            elif model == "garch_n_mfiv":
                long_horizon = selected_horizon_now(idx)

                if long_horizon is None:
                    market.record_equity()
                    save_snapshot(now)
                    continue

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
                    pd.Timedelta(days=1) <= (instrument.Expiry - now) <= pd.Timedelta(days=expiry_days)
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

            if model in ["garch", "garch_n_mfiv"]:
                # The option maturity should represent the same horizon as the variance view.
                expiry = min(
                    common_expiries,
                    key=lambda expiry_value: abs(
                        (expiry_value - now) - pd.Timedelta(hours=long_horizon)
                    )
                )
            else:
                expiry = common_expiries[0]

            expiry_calls = [instrument for instrument in valid_calls if instrument.Expiry == expiry]
            expiry_puts = [instrument for instrument in valid_puts if instrument.Expiry == expiry]

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

            atm_strike = min(common_strikes, key=lambda strike: abs(strike - last))

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

            if long_call is None or long_put is None:
                market.record_equity()
                save_snapshot(now)
                continue

            entry_risk_budget = None
            entry_stress_loss = None

            if model in ["garch", "garch_n_mfiv"]:
                equity_now = market.cash + market.option_value()
                entry_risk_budget = risk_fraction * equity_now

                # For a long straddle the premium paid is the known maximum loss at entry.
                entry_stress_loss = (
                    long_call.Price
                    + long_put.Price
                    + entry_fee(long_call)
                    + entry_fee(long_put)
                )

                if (
                    not np.isfinite(entry_risk_budget)
                    or entry_risk_budget <= 0
                    or entry_stress_loss > entry_risk_budget
                ):
                    market.record_equity()
                    save_snapshot(now)
                    continue

            before_count = len(market.holdings.instruments)

            market.buy(long_call, qty=1)
            market.buy(long_put, qty=1)

            after_count = len(market.holdings.instruments)

            if after_count - before_count == 2:
                entered_day = now.date()
                position_side = "long_straddle"
                entry_time = now

                if model == "garch_n_mfiv":
                    position_hold_hours = long_horizon
                    entry_selected_horizon = long_horizon

                    mfiv_col = f"MFIV_{long_horizon}"
                    if mfiv_col in out_full.columns:
                        mfiv_value = float(out_full.loc[now, mfiv_col])
                        entry_mfiv = mfiv_value if np.isfinite(mfiv_value) else None

                elif model == "garch":
                    position_hold_hours = garch_strike_horizon
                    entry_selected_horizon = garch_strike_horizon

                else:
                    position_hold_hours = hold_days * 24

                greek_tests.append({
                    "time": now,
                    "signal": s_now,
                    "q_lo": q_lo,
                    "q_hi": q_hi,
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
                    "rfr": float(market.time_df.loc[now, "rfr"]),
                    "risk_budget": entry_risk_budget,
                    "entry_stress_loss": entry_stress_loss
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
                if liquidate_all():
                    reset_position_state()

            market.record_equity()
            save_snapshot(now)
            continue

        # ---------------------------------------------------------
        # STRIKE HORIZON 
        # ---------------------------------------------------------
        if model == "garch":
            strike_horizon = garch_strike_horizon
            mfiv_now = None

            # sigma_hist = realised volatility from the previous H hours of spot returns.
            historical_vol_now = float(garch_historical_vol.loc[now])

            if not np.isfinite(historical_vol_now):
                market.record_equity()
                save_snapshot(now)
                continue

            spot_now = float(market.spot)

            # GARCH decides the regime; historical spot volatility defines the admissible strike range.
            lower_boundary = spot_now * np.exp(-z * historical_vol_now)
            upper_boundary = spot_now * np.exp(z * historical_vol_now)

        elif model == "garch_mfiv_avg":
            strike_horizon = None
            mfiv_now = None

            lower_boundary = last * 0.98
            upper_boundary = last * 1.02

        else:
            if entry_selected_horizon is not None:
                strike_horizon = entry_selected_horizon

            elif model in ["regime_switching_mfiv", "garch_n_mfiv"]:
                strike_horizon = selected_horizon_now(idx)

                if strike_horizon is None:
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

            mfiv_now = float(out_full.loc[now, mfiv_col])

            if not np.isfinite(mfiv_now) or mfiv_now <= 0:
                market.record_equity()
                save_snapshot(now)
                continue

            if model == "garch_n_mfiv":
                historical_vol_now = float(historical_vol[strike_horizon].loc[now])

                if not np.isfinite(historical_vol_now):
                    market.record_equity()
                    save_snapshot(now)
                    continue

                spot_now = float(market.spot)
                S_minus = spot_now * np.exp(-z * historical_vol_now)
                S_plus = spot_now * np.exp(z * historical_vol_now)

                lower_boundary = S_minus * np.exp(-z * np.sqrt(mfiv_now))
                upper_boundary = S_plus * np.exp(z * np.sqrt(mfiv_now))

            else:
                lower_boundary = last * np.exp(-z * np.sqrt(mfiv_now))
                upper_boundary = last * np.exp(z * np.sqrt(mfiv_now))


        acceptable_calls = [
            instrument
            for instrument
            in market.chain.instruments
            if (
                instrument.Right == "C"
                and instrument.Direction == "sell"
                and instrument.Strike >= upper_boundary
                and pd.Timedelta(days=1) <= instrument.Expiry - now <= pd.Timedelta(days=expiry_days)
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
            )
        ]

        # ---------------------------------------------------------
        # STRICT TWO-LEG RANGE FILL
        # ---------------------------------------------------------
        selected_calls = []
        selected_puts = []
        entry_risk_budget = None
        entry_stress_loss = None
        selected_pair_scores = []

        if model == "garch":
            target_expiry_hours = garch_strike_horizon
        elif model == "garch_n_mfiv":
            target_expiry_hours = strike_horizon
        else:
            target_expiry_hours = expiry_days * 24

        common_expiries = sorted(
            {
                instrument.Expiry for instrument in acceptable_calls
            }
            & {
                instrument.Expiry for instrument in acceptable_puts
            },
            key=lambda expiry_value: abs(
                (expiry_value - now) - pd.Timedelta(hours=target_expiry_hours)
            )
        )

        if model in ["garch", "garch_n_mfiv"]:
            # These traders require one common expiry matched to the horizon of their variance view.
            if not common_expiries:
                market.record_equity()
                save_snapshot(now)
                continue

            expiry = common_expiries[0]

            expiry_calls = sorted(
                [
                    instrument for instrument in acceptable_calls
                    if instrument.Expiry == expiry
                ],
                key=lambda instrument: instrument.Strike
            )[:quantity]

            expiry_puts = sorted(
                [
                    instrument for instrument in acceptable_puts
                    if instrument.Expiry == expiry
                ],
                key=lambda instrument: instrument.Strike,
                reverse=True
            )[:quantity]

            if not expiry_calls or not expiry_puts:
                market.record_equity()
                save_snapshot(now)
                continue

            spot_now = float(market.spot)
            dS_lower = lower_boundary - spot_now
            dS_upper = upper_boundary - spot_now

            equity_now = market.cash + market.option_value()
            entry_risk_budget = risk_fraction * equity_now

            if not np.isfinite(entry_risk_budget) or entry_risk_budget <= 0:
                market.record_equity()
                save_snapshot(now)
                continue

            call_stats = []
            for call in expiry_calls:
                greeks = option_greeks(now, call)
                if greeks is not None:
                    call_stats.append((call, greeks))

            put_stats = []
            for put in expiry_puts:
                greeks = option_greeks(now, put)
                if greeks is not None:
                    put_stats.append((put, greeks))

            pair_candidates = []

            for call, call_greeks in call_stats:
                for put, put_greeks in put_stats:
                    pair_delta = call_greeks["delta"] + put_greeks["delta"]
                    pair_gamma = call_greeks["gamma"] + put_greeks["gamma"]

                    pnl_lower = (
                        pair_delta * dS_lower
                        + 0.5 * pair_gamma * dS_lower ** 2
                    )

                    pnl_upper = (
                        pair_delta * dS_upper
                        + 0.5 * pair_gamma * dS_upper ** 2
                    )

                    # stress_loss = delta-gamma loss if spot reaches either entry risk boundary.
                    stress_loss = max(0.0, -pnl_lower, -pnl_upper)

                    net_credit = (
                        call.Price
                        + put.Price
                        - entry_fee(call)
                        - entry_fee(put)
                    )

                    if stress_loss <= 0 or net_credit <= 0:
                        continue

                    pair_candidates.append({
                        "call": call,
                        "put": put,
                        "delta": pair_delta,
                        "gamma": pair_gamma,
                        "stress_loss": stress_loss,
                        "net_credit": net_credit,
                        # Premium earned per dollar of model-consistent boundary risk.
                        "score": net_credit / stress_loss
                    })

            pair_candidates = sorted(
                pair_candidates,
                key=lambda pair: (pair["score"], pair["net_credit"]),
                reverse=True
            )

            used_calls = set()
            used_puts = set()
            total_delta = 0.0
            total_gamma = 0.0

            for pair in pair_candidates:
                if len(selected_calls) >= quantity:
                    break

                if pair["call"].Symbol in used_calls or pair["put"].Symbol in used_puts:
                    continue

                trial_delta = total_delta + pair["delta"]
                trial_gamma = total_gamma + pair["gamma"]

                trial_pnl_lower = (
                    trial_delta * dS_lower
                    + 0.5 * trial_gamma * dS_lower ** 2
                )

                trial_pnl_upper = (
                    trial_delta * dS_upper
                    + 0.5 * trial_gamma * dS_upper ** 2
                )

                trial_stress_loss = max(
                    0.0,
                    -trial_pnl_lower,
                    -trial_pnl_upper
                )

                if trial_stress_loss > entry_risk_budget:
                    continue

                selected_calls.append(pair["call"])
                selected_puts.append(pair["put"])
                selected_pair_scores.append(pair["score"])

                used_calls.add(pair["call"].Symbol)
                used_puts.add(pair["put"].Symbol)

                total_delta = trial_delta
                total_gamma = trial_gamma
                entry_stress_loss = trial_stress_loss

            if not selected_calls or not selected_puts:
                market.record_equity()
                save_snapshot(now)
                continue

        else:
            # Legacy/control models keep their original contract-selection behaviour.
            for expiry in common_expiries:
                expiry_calls = sorted(
                    [
                        instrument for instrument in acceptable_calls
                        if instrument.Expiry == expiry
                    ],
                    key=lambda instrument: instrument.Strike
                )

                expiry_puts = sorted(
                    [
                        instrument for instrument in acceptable_puts
                        if instrument.Expiry == expiry
                    ],
                    key=lambda instrument: instrument.Strike,
                    reverse=True
                )

                if expiry_calls and expiry_puts:
                    selected_calls = expiry_calls[:quantity]
                    selected_puts = expiry_puts[:quantity]
                    break

            if not selected_calls or not selected_puts:
                if acceptable_calls:
                    call_expiry = min(
                        {
                            instrument.Expiry
                            for instrument in acceptable_calls
                        },
                        key=lambda expiry_value: abs(
                            (expiry_value - now) - pd.Timedelta(days=expiry_days)
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
                            instrument.Expiry for instrument in acceptable_puts
                        },
                        key=lambda expiry_value: abs(
                            (expiry_value - now) - pd.Timedelta(days=expiry_days)
                        )
                    )

                    selected_puts = sorted(
                        [
                            instrument for instrument in acceptable_puts
                            if instrument.Expiry == put_expiry
                        ],
                        key=lambda instrument: instrument.Strike,
                        reverse=True
                    )[:quantity]

            pair_count = min(len(selected_calls), len(selected_puts), quantity)

            if pair_count == 0:
                market.record_equity()
                save_snapshot(now)
                continue

            selected_calls = selected_calls[:pair_count]
            selected_puts = selected_puts[:pair_count]

        before_call_count = call_count
        before_put_count = put_count

        for instrument in selected_calls:
            market.sell(instrument, qty=1)

        for instrument in selected_puts:
            market.sell( instrument, qty=1 )

        holdings_after = list(market.holdings.instruments)

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

        if calls_added <= 0 or puts_added <= 0 or calls_added != puts_added:
            if liquidate_all():
                reset_position_state()
            market.record_equity()
            save_snapshot(now)
            continue

        entered_day = now.date()
        position_side = "short_strangle"
        entry_time = now

        if model == "garch_n_mfiv":
            position_hold_hours = strike_horizon
        else:
            position_hold_hours = hold_days * 24

        entry_selected_horizon = (strike_horizon)

        entry_mfiv = mfiv_now

        entry_lower_boundary = (lower_boundary)

        entry_upper_boundary = (upper_boundary)

        greek_tests.append({
            "time": now,
            "signal": s_now,
            "q_lo": q_lo,
            "q_hi": q_hi,
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
            "rfr": float(market.time_df.loc[now, "rfr"]),
            "risk_budget": entry_risk_budget,
            "entry_stress_loss": entry_stress_loss,
            "pair_scores": selected_pair_scores
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