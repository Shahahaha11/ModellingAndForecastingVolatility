### NEW 9 SEP 28 edits
# 8: TCN attempts to reach 10 matched quantity immediately, then at each of the next 3 hourly observations if the same TCN action remains active, without resetting the original 24-hour holding clock.
# 9 : YEAR-END LIQUIDATION WINDOW block 2 replaced to use new get_maintenance_margin from market.py

import pandas as pd
import numpy as np
from srcB.helpers import *

def testing_execute_trading_strategy(
    bt_year, bt_quarter, model, outputs_by_year, market, trader_cache,
    run_by="year",
    quantity=10, tcn_score_threshold=0.0, forward_hours=24
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
        "tcn_mfiv"
    ]

    if model not in VALID_MODELS:
        raise ValueError(f"model must be one of {VALID_MODELS}")

    tcn_action_series = out_full["tcn_action"].astype("string").reindex(now_list)
    tcn_best_score_series = pd.to_numeric( out_full["tcn_best_score"], errors="coerce").reindex(now_list)
    
    position_side = None
    entry_time = None
    position_hold_hours = None

    entry_selected_horizon = None
    entry_mfiv = None

    def reset_position_state():
        nonlocal position_side
        nonlocal entry_time
        nonlocal position_hold_hours
        nonlocal entry_selected_horizon
        nonlocal entry_mfiv

        position_side = None
        entry_time = None
        position_hold_hours = None

        entry_selected_horizon = None
        entry_mfiv = None

    def liquidate_all(liquidation=False):
        for instrument in list(market.holdings.instruments):
            key = (
                instrument.Strike,
                instrument.Expiry,
                instrument.Right,
                instrument.Direction
            )
            market.liquidate(
                key,
                instrument,
                liquidation=liquidation
            )
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

        if market.cash < market.get_maintenance_margin():
            if liquidate_all(liquidation=True):
                reset_position_state()

            market.record_equity()
            save_snapshot(now)
            continue
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

        if not has_position and entry_time is not None:
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

            if should_exit:

                if liquidate_all():
                    reset_position_state()

                market.record_equity()
                save_snapshot(now)
                continue

        if complete_position:

            tcn_position_qty = (call_count if short_position else long_call_count)

            if (tcn_position_qty >= float(quantity)
                or entry_time is None
                or now > entry_time + pd.Timedelta(hours=3)):
                market.record_equity()
                save_snapshot(now)
                continue
            
        # ---------------------------------------------------------
        # TCN ENTRY
        # ---------------------------------------------------------
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
            call_margin, _ = market.option_margin(selected_call, pair_qty)
            put_margin, _ = market.option_margin(selected_put, pair_qty)
            new_margin = call_margin + put_margin

            trade_exists = (
                call_exists
                and put_exists
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

            position_side = action_now
            entry_time = now
            position_hold_hours = forward_hours
            entry_selected_horizon = forward_hours

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

    liquidate_all()
    market.record_equity()
    save_snapshot(market.now)

    for test in greek_tests:
        test["entry_equity"] = next(
            s["equity"] for s in snapshots
            if s["time"] == test["time"]
        )

    return snapshots, greek_tests