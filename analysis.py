# SEARCH WORDS :
# FINAL_READY_OUTPUTS_BY_YEAR
# YEARLY_TCN_ESTIMATION
# YEARLY_MFIV_GARCH_ESTIMATION
# TESTING_MAIN_WITH_CACHE
# TCN_SCORE_CALIBRATION_DIAGNOSTIC
## TCN_SCORE_THRESHOLD_CALIBRATION
### TCN_SCORE_THRESHOLD_TRADER_CALIBRATION
import pandas as pd
import numpy as np
from math import sqrt, pi, exp
from arch import arch_model
import yfinance as yf 
from scipy.optimize import minimize
import warnings
from arch.__future__ import reindexing
from arch.utility.exceptions import ConvergenceWarning
# from srcA.msGarch import msGARCH
# from srcA.metrics import metrics
# from srcA.msGarch import msGARCHIV
from tqdm import tqdm

from xgboost import XGBRegressor
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="y is poorly scaled")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


import os, sys
sys.path.insert(0, "/Users/shah/CODE_BOOK_4/THESIS_WORKING/THESIS_2")
sys.path.insert(0, "/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/MFIV_calibration")
os.chdir("/Users/shah/CODE_BOOK_4/THESIS_WORKING/THESIS_2")
sys.modules.pop("srcB", None)

### NOTE :
# # COMES FROM MODELLING.ipynb
import pickle 
with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/outputs_by_year/outputs_by_year_03.pkl", "rb") as f:
    outputs_by_year = pickle.load(f)


# IMPORTS FROM SRC
from srcB.market import *
from srcB.trader_testing import testing_execute_trading_strategy
from srcB.model_testing import regime_switching_mfiv
from srcB.helpers import *

# READ D_BTC AND DF
from pathlib import Path
DATA_DIR = Path("../data/rawData")
YEARS = [2019, 2020, 2021]
d_btc_raw_by_year = {}
df_raw_by_year = {}
for y in YEARS:
    d_btc_raw_by_year[y] = pd.read_parquet(DATA_DIR / f"btc_underlying_minutely_{y}.parquet")
    df_raw_by_year[y]    = pd.read_parquet(DATA_DIR / f"derivative_data_{y}.parquet")

# SET INDEX TO DATETIME
for y in YEARS:
    d_btc_raw_by_year[y] = d_btc_raw_by_year[y].set_index(pd.to_datetime(d_btc_raw_by_year[y]["date_time"]))
    df_raw_by_year[y] = df_raw_by_year[y].set_index(pd.to_datetime(df_raw_by_year[y]["date_time"]))



# ==================================================================================================================
# ======================================= LOAD XQ_BY_YEAR USING X_BY_YEAR ==========================================
# ==================================================================================================================
YEARS = [2019, 2020, 2021]

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/x_by_year_fut/x_by_year.pkl", "rb") as f:
    X_by_year = pd.read_pickle(f)

# X_by_year[2021]["rt"] # CHECKING DELETE
for y in YEARS:
    X_by_year[y] = X_by_year[y].copy()
    X_by_year[y]["rv"] = X_by_year[y]["rt"] ** 2

Xq_by_year = {}
YEARS = [2019, 2020, 2021]
for y in YEARS:
    Xq_by_year[y] = {}
    Xq_by_year[y]["q1"] = X_by_year[y].loc[f"{y}-01-01" : f"{y}-03-31 23:59:59"]
    Xq_by_year[y]["q2"] = X_by_year[y].loc[f"{y}-04-01" : f"{y}-06-30 23:59:59"]
    Xq_by_year[y]["q3"] = X_by_year[y].loc[f"{y}-07-01" : f"{y}-09-30 23:59:59"]
    Xq_by_year[y]["q4"] = X_by_year[y].loc[f"{y}-10-01" : f"{y}-12-31 23:59:59"]

# =============================================================================================================
# ============== MFIV SCALING ============= SCALING ONLY ON XQ  =================================================
# ================== CONVERTS ANNUALIZED MFIV -> CUMULATIVE H-HOUR DECIMAL VARIANCE ===========================

MFIVS = list(range(2, 577))
QUARTERS = ["q1", "q2", "q3", "q4"]

for y in YEARS:
    for q in QUARTERS:
        for H in MFIVS:
            Xq_by_year[y][q][f"MFIV_{H}"] *= (H / 8760.0)

# ==================================================================================================================
# ============================================= BORROW RFR COLUMN  =================================================
# ==================================================================================================================
import pickle
with open ("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/ready_data/d_btc_by_year_all.pkl", "rb") as f :
    d_btc_by_year = pickle.load(f)

YEARS = [2019, 2020, 2021]
for y in YEARS:
    d_btc_raw_by_year[y]["rfr"] = d_btc_by_year[y]["rfr"].reindex(d_btc_raw_by_year[y].index, method="ffill")

# =============================================================================================================
# ======================================= CHECKPOINT 1. RUN UNTIL HERE ========================================
# =============================================================================================================











# # =============================================================================================================
# # ======================================= PREPARE EMPTY DICTS =================================================
# # =============================================================================================================

# YEARS = [2019, 2020, 2021]
# QUARTERS = ["q1", "q2", "q3", "q4"]

# idx_by_year = {y : {} for y in YEARS}
# f_rollB_by_year = {y: {} for y in YEARS}
# rolling_forecasts =  {y: {} for y in YEARS}
# rolling_residuals = {y: {} for y in YEARS}
# y_pred_by_year = {y: {} for y in YEARS}
# y_true_by_year = {y: {} for y in YEARS}



# # ==================================================================================================================
# # ======================================= YEARLY_TCN_ESTIMATION  ====================================================
# # ==================================================================================================================
sys.modules.pop("srcB.tcn_model", None)
sys.modules.pop("srcB.tcn_model_v2", None)
from srcB.tcn_model_v2 import *
YEARS = [2019, 2020, 2021]
QUARTERS = ["q1", "q2", "q3", "q4"]
tcn_bundle_by_year = {}
tcn_targets_by_year = {}
tcn_output_by_year = {}

for y in YEARS:
    X_df = pd.concat([Xq_by_year[y][q] for q in QUARTERS]).sort_index().copy()
    garch_df = pd.concat([outputs_by_year[y][q]["garch"] for q in QUARTERS]).sort_index()
    X_df["garch"] = garch_df.reindex(X_df.index)
    tcn_output, tcn_bundle, tcn_targets = tcn_mfiv(
        X_df=X_df,
        option_df=df_raw_by_year[y],
        time_df=d_btc_raw_by_year[y],
        sequence_length=168,
        forward_hours=24,
        delta_target=0.20
    )
    tcn_output_by_year[y] = tcn_output
    tcn_bundle_by_year[y] = tcn_bundle
    tcn_targets_by_year[y] = tcn_targets

    for q in QUARTERS:
        q_index = Xq_by_year[y][q].index
        tcn_q = tcn_output.reindex(q_index)
        outputs_by_year[y][q] = pd.concat( [outputs_by_year[y][q], tcn_q], axis=1 )

# =============================================================================================================
# ======================================= SAVE TCN BUNDLES ====================================================
# =============================================================================================================
os.makedirs("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/TCN", exist_ok = True)
import pickle
with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/TCN/tcn_bundle_by_year.pkl", "wb") as f:
    pickle.dump(tcn_bundle_by_year, f)

print("TCN bundles saved:", list(tcn_bundle_by_year.keys()))


##==================================================================================================================
##======================================= YEARLY_MFIV_GARCH_ESTIMATION  ============================================
##==================================================================================================================
# YEARLY ESTIMATION
# YEARS = [2019, 2020, 2021]
# QUARTERS = ["q1", "q2", "q3", "q4"]

# # THE_COLS = [f"MFIV_{h}" for h in range(2, 577)]
# HORIZONS = [12, 24, 72, 168, 336]
# THE_COLS = [f"MFIV_{h}" for h in HORIZONS]

# for y in YEARS:
#     X_df = pd.concat([Xq_by_year[y][q] for q in QUARTERS]).sort_index()
#     f_rollB, y_true, idx, regime_state, selected_horizon = regime_switching_mfiv(
#         X_df,
#         THE_COLS,
#         score_window=168
#     )
#     full_forecast = pd.Series(f_rollB, index=X_df.index[idx])
#     full_true = pd.Series(y_true, index=X_df.index[idx])
#     full_regime = pd.Series(regime_state, index=X_df.index[idx])
#     full_horizon = pd.Series(selected_horizon, index=X_df.index[idx])
#     for q in QUARTERS:
#         q_index = Xq_by_year[y][q].index
#         f_q = full_forecast.reindex(q_index).dropna()
#         y_q = full_true.reindex(f_q.index)
#         regime_q = full_regime.reindex(f_q.index)
#         horizon_q = full_horizon.reindex(f_q.index)
#         idx_q = Xq_by_year[y][q].index.get_indexer(f_q.index)
#         idx_by_year[y][q] = idx_q
#         f_rollB_by_year[y][q] = f_q.to_numpy()
#         rolling_forecasts[y][q] = f_q.to_numpy()
#         rolling_residuals[y][q] = (y_q - f_q).to_numpy()
#         y_pred_by_year[y][q] = f_q.to_numpy()
#         y_true_by_year[y][q] = y_q.to_numpy()
#         outputs_by_year[y][q] = pd.concat([outputs_by_year[y][q], pd.DataFrame({"regime_switching_mfiv": f_q, "regime_state": regime_q, "selected_horizon" : horizon_q})], axis = 1)

# ==================================================================================================================
# SAVE THIS OUTPUTS
# with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/outputs_by_year_analysis_fut.pkl", "wb") as f:
#     pickle.dump(outputs_by_year, f)
# ==================================================================================================================

# ===========================================================================================================
# ==================================== OBJECTS YOU NEED BEFORE RUNNING BELOW ================================
# ============= TEMPORARY ADD ON, MUST REMOVE, WITHOUT CHANGE, SAVE THE PREPARED OUTPUTS_BY_YEAR ============
# ===========================================================================================================
### temporary load 
### this below one only has one extra thing. it has only 
# import pickle 
# with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/outputs_by_year_analysis_fut.pkl", "rb") as f:
#     outputs_by_year_fut = pickle.load(f)

# # with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/outputs_by_year/outputs_by_year_03.pkl", "rb") as f:
# #     outputs_by_year_03 = pickle.load(f)

# ### dump this prepared data set above CAREFULLY LATER
# for y in YEARS:
#     for q in QUARTERS:
#         outputs_by_year[y][q] = pd.concat(
#             [
#                 outputs_by_year[y][q],
#                 outputs_by_year_fut[y][q][
#                     ["selected_horizon", "regime_state", "regime_switching_mfiv"]
#                 ]
#             ],
#             axis=1
#         )
# ### REMOVE DUPLICATES FRM THAT CONCAT
# for y in YEARS:
#     for q in QUARTERS:
#         outputs_by_year[y][q] = outputs_by_year[y][q].loc[
#             :, ~outputs_by_year[y][q].columns.duplicated(keep="last")
#         ]

#################################### FINAL_READY_OUTPUTS_BY_YEAR AS 04 ####################################
# import pickle
# with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/outputs_by_year/outputs_by_year_04.pkl", "wb") as f:
#     pickle.dump(outputs_by_year, f)













# =============================================================================================================
# ======================================= CHECKPOINT 2. RUN FROM HERE ==================================================================================================
# =============================================================================================================

import pickle
with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/outputs_by_year/outputs_by_year_04.pkl", "rb") as f:
    outputs_by_year = pickle.load(f)
# ==================================== BORROW MFIV COLS FROM XQ_BY_YEAR ================================
# ===========================================================================================================

MFIVS = list(range(2, 577))
QUARTERS = ["q1", "q2", "q3", "q4"]
mfiv_cols = [f"MFIV_{H}" for H in MFIVS]

for y in YEARS:
    for q in QUARTERS:
        outputs_by_year[y][q].drop(columns=mfiv_cols, inplace=True)

for y in YEARS:
    for q in QUARTERS:
        outputs_by_year[y][q] = pd.concat(
            [outputs_by_year[y][q], Xq_by_year[y][q][mfiv_cols]],
            axis=1
        )

# # IMPORTS FROM SRC
from srcB.market import *
from srcB.trader_testing import testing_execute_trading_strategy
from srcB.helpers import *

# ==================================================================================================================
# ======================================= CACHE FOR WARMUP ==========================================================
# ==================================================================================================================
CACHE_YEARS = [2019, 2020, 2021]
RUN_BY = "year"
if RUN_BY == "year":
    trader_cache = {
        y: pd.concat(
            [
                outputs_by_year[yy][q]
                for yy in CACHE_YEARS
                if yy <= y
                for q in QUARTERS
            ]
        ).sort_index()
        for y in CACHE_YEARS
    }

elif RUN_BY == "quarter":
    trader_cache = {
        (y, q): pd.concat(
            [
                outputs_by_year[yy][qq]
                for yy in CACHE_YEARS
                for qq in QUARTERS
                if (
                    yy < y
                    or (yy == y and QUARTERS.index(qq) <= QUARTERS.index(q))
                )
            ]
        ).sort_index()
        for y in CACHE_YEARS
        for q in QUARTERS
    }
if RUN_BY == "year":
    PERIODS = YEARS
else:
    PERIODS = [(y, q) for y in YEARS for q in QUARTERS]
    


# ==================================================================================================================
 # ======================================= TESTING_MAIN_WITH_CACHE  ==========================================================
# ==================================================================================================================
# sys.modules.pop("srcB", None)

sys.modules.pop("srcB.trader_testing_tcn", None)
from srcB.trader_testing_tcn import testing_execute_trading_strategy
YEARS = [2019, 2020, 2021]
MODELS = [
    "garch",
    "garch_n_mfiv",
    "garch_mfiv_avg",
    "tcn_mfiv"
]
RUN_BY = "year"
strategy_results = {}
for YEAR in YEARS:
    for MODEL in MODELS:
        market = Market(
            option_df=df_raw_by_year[YEAR],
            time_df=d_btc_raw_by_year[YEAR],
            starting_cash=100_000,
            im_multiplier=0.15,
            mm_buffer_rate=0.05,
        )
        greek_tests = []
        yearly_snapshots = []
        if RUN_BY == "year":
            periods = [None]
        else:
            periods = ["q1", "q2", "q3", "q4"]
            
        for q in periods:
            snapshots, greek_tests_q = testing_execute_trading_strategy(
                bt_year=YEAR,
                bt_quarter=q,
                model=MODEL,
                outputs_by_year=outputs_by_year,
                market=market,
                trader_cache=trader_cache,
                run_by=RUN_BY,
                quantity=10,
                expiry_days=30,
                hold_days=7,
                lookback_hours=168,
                mean_hours=12,
                ql=0.10,
                qh=0.90,
                z=1.645
            )

            greek_tests.extend(greek_tests_q)
            yearly_snapshots.extend(snapshots)
        history_key = f"{MODEL}_{str(YEAR)[2:]}"
        equity_curve = market.get_equity_history()
        strategy_results[history_key] = {
            "equity_curve": equity_curve,
            "snapshots": yearly_snapshots,
            "greek_tests": greek_tests,
            "orders": list(market.orders),
            "transactions": list(market.transactions),
            "final_cash": market.cash,
            "required_margin": market.get_total_required_margin(),
            "free_liquidity": market.get_free_liquidity(),
        }
print(YEAR, MODEL)

# ==================================================================================================================
print(len(greek_tests))
print(greek_tests[:1])
print(pd.Series([x["action"] for x in market.transactions]).value_counts())
print("outputs:", "tcn_action" in outputs_by_year[2019]["q1"].columns)
print("cache:", "tcn_action" in trader_cache[2019].columns)
[c for c in trader_cache[2019].columns if c.startswith("tcn")]

## ==================================================================================================================
## ============================================= SAVE DICT  ========================================================
## ==================================================================================================================
# import sys, os
# os.makedirs("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/data_strategy_results", exist_ok = True)
# import pickle 
# with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/data_strategy_results/strategy_results.pkl", "wb") as f:
#     pickle.dump(strategy_results, f)

# # LOAD BACK 
# with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/data_strategy_results/strategy_results.pkl", "rb") as f:
#     strategy_results = pickle.load(f)

## ==================================================================================================================
## ============================================= SHARPEE  ===========================================================
## ==================================================================================================================
YEARS = [2019, 2020, 2021]
MODELS = [
    "garch",
    "garch_n_mfiv",
    "garch_mfiv_avg",
    "tcn_mfiv"
]
rows = []

for y in YEARS:
    for model in MODELS:

        history_key = f"{model}_{str(y)[2:]}"
        snapshots = strategy_results[history_key]["snapshots"]

        equity = pd.Series( [s["equity"] for s in snapshots], index=[s["time"] for s in snapshots] )

        returns = equity.pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(24 * 365)

        rows.append({ "year": y, "model": model, "sharpe": sharpe })

sharpe_table = pd.DataFrame(rows)

print(sharpe_table)


# Is portfolio liquidated at the end of year YES
MODELS = ["garch", "garch_mfiv", "garch_n_mfiv", "garch_mfiv_avg", "regime_switching_mfiv"]
rows = []
for y in YEARS:
    for model in MODELS:
        if model.startswith("garch"):
            history_key = f"{model}_{str(y)[2:]}"
            rows.append({
                "year" : y,
                "model" : model,
                "no_position" : "yes" if len(strategy_results[history_key]["snapshots"][-1]["positions"]) == 0 else "no"
            })
position_free = pd.DataFrame(rows)
print(position_free)


# ==================================================================================================================
# ============================================= PLOT EQUITY  =======================================================
# ==================================================================================================================
import matplotlib.pyplot as plt
# MODELS = ["garch", "garch_mfiv", "garch_n_mfiv", "garch_mfiv_avg", "regime_switching_mfiv"]
MODELS = [
    "garch",
    "garch_n_mfiv",
    "garch_mfiv_avg",
    "tcn_mfiv"
]
PLOT_YEAR = 2019
plt.figure(figsize=(12, 6))
for MODEL in MODELS:
    plot_key = f"{MODEL}_{str(PLOT_YEAR)[2:]}"
    result = strategy_results[plot_key]
    eq = pd.DataFrame(result["snapshots"]).drop_duplicates("time").sort_values("time")
    plt.plot(eq["time"], eq["equity"], label=MODEL)
plt.axhline(100_000, linestyle="--")
plt.title(f"Equity curve — {PLOT_YEAR}")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# ==================================================================================================================
# ============================================= PLOT DELTA  ========================================================
# ==================================================================================================================
import matplotlib.pyplot as plt
PLOT_YEAR = 2020
greek_tests = strategy_results[f"garch_n_mfiv_{str(PLOT_YEAR)[2:]}"]["greek_tests"]
M = 1
N = 5
for x in greek_tests[M:N]:
    call_strikes = [i.Strike for i in x["selected_calls"]]
    put_strikes = [i.Strike for i in x["selected_puts"]]
    call_delta = [g["delta"] for g in x["call_greeks"]]
    put_delta = [g["delta"] for g in x["put_greeks"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(call_strikes, call_delta, color="green", label="Calls")
    ax.scatter(put_strikes, put_delta, color="red", label="Puts")
    ax.set_title(str(x["time"]))
    ax.set_xlabel("Strike")
    ax.set_ylabel("Delta")
    ax.legend()
    plt.show()

# ========================================================================================================================================
# ====================================================== STRATEGY DIAGNOSTICS ============================================================
# ========================================================================================================================================
y = 2021 ; the_model = "garch_n_mfiv"
# the_model = "garch_n_mfiv"
position_side = "short_strangle"
tx = pd.DataFrame(strategy_results[f"{the_model}_{str(y)[2:]}"]["transactions"]).sort_values("time")
open_actions = {"buy_to_open", "sell_to_open"}
close_actions = {"buy_to_close", "sell_to_close", "liquidate", "expiry_settlement"}
trades = []
open_pos = {}
for _, r in tx.iterrows():
    sym = r["symbol"]
    if r["action"] in open_actions:
        open_pos[sym] = {
            "entry_time": r["time"],
            "right": r["right"],
            "direction": r["held_direction"],
            "pnl": float(r["cash_change"])
        }
    elif r["action"] in close_actions and sym in open_pos:
        open_pos[sym]["pnl"] += float(r["cash_change"])
        open_pos[sym]["exit_time"] = r["time"]
        trades.append({"symbol": sym, **open_pos.pop(sym)})
legs = pd.DataFrame(trades)
groups = []
for entry_time, g in legs.groupby("entry_time"):
    side = "short_strangle" if (g["direction"] == "sell").all() else "long_straddle"
    if side != position_side:
        continue
    call_pnl = g.loc[g["right"] == "C", "pnl"].sum()
    put_pnl = g.loc[g["right"] == "P", "pnl"].sum()
    groups.append({
        "entry": entry_time,
        "exit": g["exit_time"].max(),
        "side": side,
        "call_pnl": call_pnl,
        "put_pnl": put_pnl,
        "total_pnl": call_pnl + put_pnl,
        "call_result": "PROFIT" if call_pnl > 0 else "LOSS",
        "put_result": "PROFIT" if put_pnl > 0 else "LOSS" if put_pnl < 0 else "FLAT",
        "position_result": "PROFIT" if call_pnl + put_pnl > 0 else "LOSS"
    })
result = pd.DataFrame(groups)

print(result.to_string(index=False))
print("\nPositions:", len(result))
print("Profitable positions:", (result["total_pnl"] > 0).sum())
print("Losing positions:", (result["total_pnl"] < 0).sum())
print("Call legs profit:", (result["call_pnl"] > 0).sum())
print("Call legs loss:", (result["call_pnl"] < 0).sum())
print("Put legs profit:", (result["put_pnl"] > 0).sum())
print("Put legs loss:", (result["put_pnl"] < 0).sum())

# ========================================================================================================================================
# ========================================== ALL STRATEGY DIAGNOSTICS  ===================================================================
# ========================================================================================================================================
# position_side = "long_straddle"
position_side = "short_strangle"
all_groups = []
for y in YEARS:
    for the_model in MODELS:
        tx = pd.DataFrame(strategy_results[f"{the_model}_{str(y)[2:]}"]["transactions"]).sort_values("time")
        open_actions = {"buy_to_open", "sell_to_open"}
        close_actions = {"buy_to_close", "sell_to_close", "liquidate", "expiry_settlement"}
        trades = []
        open_pos = {}
        for _, r in tx.iterrows():
            sym = r["symbol"]
            if r["action"] in open_actions:
                open_pos[sym] = {
                    "entry_time": r["time"],
                    "right": r["right"],
                    "direction": r["held_direction"],
                    "pnl": float(r["cash_change"])
                }
            elif r["action"] in close_actions and sym in open_pos:
                open_pos[sym]["pnl"] += float(r["cash_change"])
                open_pos[sym]["exit_time"] = r["time"]

                trades.append({
                    "symbol": sym,
                    **open_pos.pop(sym)
                })
        legs = pd.DataFrame(trades)
        for entry_time, g in legs.groupby("entry_time"):
            side = (
                "short_strangle"
                if (g["direction"] == "sell").all()
                else "long_straddle"
            )
            if side != position_side:
                continue
            call_pnl = g.loc[g["right"] == "C", "pnl"].sum()
            put_pnl = g.loc[g["right"] == "P", "pnl"].sum()
            all_groups.append({
                "year": y,
                "model": the_model,
                "entry": entry_time,
                "exit": g["exit_time"].max(),
                "side": side,
                "call_pnl": call_pnl,
                "put_pnl": put_pnl,
                "total_pnl": call_pnl + put_pnl,
                "call_result": "PROFIT" if call_pnl > 0 else "LOSS",
                "put_result": "PROFIT" if put_pnl > 0 else "LOSS" if put_pnl < 0 else "FLAT",
                "position_result": "PROFIT" if call_pnl + put_pnl > 0 else "LOSS"
            })
result_all = pd.DataFrame(all_groups)
print(result_all.to_string(index=False))
summary = (
    result_all.groupby(["year", "model"])
    .agg(
        positions=("total_pnl", "size"),
        profitable=("total_pnl", lambda x: (x > 0).sum()),
        losing=("total_pnl", lambda x: (x < 0).sum()),
        total_pnl=("total_pnl", "sum"),
        call_profit=("call_pnl", lambda x: (x > 0).sum()),
        call_loss=("call_pnl", lambda x: (x < 0).sum()),
        put_profit=("put_pnl", lambda x: (x > 0).sum()),
        put_loss=("put_pnl", lambda x: (x < 0).sum())
    )
    .reset_index()
)
print("\nSUMMARY")
print(summary.to_string(index=False))

# ========================================================================================================================================
# ========================================== TCN SPECIFIC (TRADER DIAGNOSTICS)============================================
# ========================================================================================================================================
MODEL = "tcn_mfiv"

all_trades = []
for y in YEARS:
    tx = pd.DataFrame(strategy_results[f"{MODEL}_{str(y)[2:]}"]["transactions"]).sort_values("time")

    open_actions = {"buy_to_open", "sell_to_open"}
    close_actions = {"buy_to_close", "sell_to_close", "liquidate", "expiry_settlement"}

    trades = []
    open_pos = {}

    for _, r in tx.iterrows():
        sym = r["symbol"]
        if r["action"] in open_actions:
            open_pos[sym] = {
                "entry_time": r["time"],
                "right": r["right"],
                "strike": r["strike"],
                "direction": r["held_direction"],
                "pnl": float(r["cash_change"])
            }

        elif r["action"] in close_actions and sym in open_pos:
            open_pos[sym]["pnl"] += float(r["cash_change"])
            open_pos[sym]["exit_time"] = r["time"]
            trades.append({
                "symbol": sym,
                **open_pos.pop(sym)
            })

    legs = pd.DataFrame(trades)
    for entry_time, g in legs.groupby("entry_time"):
        direction = "short" if (g["direction"] == "sell").all() else "long"
        structure = "straddle" if g["strike"].nunique() == 1 else "strangle"
        side = f"{direction}_{structure}"
        call_pnl = g.loc[g["right"] == "C", "pnl"].sum()
        put_pnl = g.loc[g["right"] == "P", "pnl"].sum()
        total_pnl = call_pnl + put_pnl

        all_trades.append({
            "year": y,
            "entry": entry_time,
            "exit": g["exit_time"].max(),
            "side": side,
            "call_pnl": call_pnl,
            "put_pnl": put_pnl,
            "total_pnl": total_pnl,
            "position_result": "PROFIT" if total_pnl > 0 else "LOSS"
        })

tcn_trades = pd.DataFrame(all_trades)
print(tcn_trades.to_string(index=False))

summary = (
    tcn_trades.groupby(["year", "side"])
    .agg(
        positions=("total_pnl", "size"),
        profitable=("total_pnl", lambda x: (x > 0).sum()),
        losing=("total_pnl", lambda x: (x < 0).sum()),
        total_pnl=("total_pnl", "sum"),
        avg_pnl=("total_pnl", "mean")
    )
    .reset_index()
)
print("\nSUMMARY")
print(summary.to_string(index=False))

# ======================= SPEECH ON DIAGNOSTICS ============================================
# The TCN is using all four actions, but very unevenly. 
# In 2019 it mostly uses long straddles and short straddles, and both lose overall.
# In 2020 it is overwhelmingly concentrated in long straddles, which again lose overall. 
# In 2021 the model shifts much more heavily toward short straddles, while long straddles become extremely profitable. 
# The strong 2021 Sharpe is therefore not coming from one action alone: 
# long straddles contribute about +31k, 
# short straddles about +13k, 
# short strangles about +4.9k, 
# while long strangles lose about -2.7k. 

# The most interesting part is the long straddle result in 2021: 
# it wins only about half the time, but still produces very large total PnL, which is consistent with the convex payoff of long volatility positions where a few large gains can dominate many smaller losses. 
# The main caution is that the raw dollar PnL is not directly comparable across 2019, 2020, and 2021 because BTC price levels differ materially, 
# so the next useful check is to normalize realized trade PnL by entry spot,
# matching the way the TCN targets were constructed. 


# # ============================================= PHASE 2 =====================================================
# # ======================================= DIAGNOSTICS FOR TCN ===============================================
# # ===========================================================================================================
## =============================================================================================================
## ============================== TCN FITTED TARGET-SPACE PERFORMANCE ==========================================
## =============================================================================================================

target_cols = [
    "target_long_straddle",
    "target_short_straddle",
    "target_long_strangle",
    "target_short_strangle"
]
rows = []
for y in YEARS:
    test = pd.concat([
        tcn_output_by_year[y][["tcn_action"]],
        tcn_targets_by_year[y][target_cols]
    ], axis=1, sort = False).dropna()
    realized = pd.Series(0.0, index=test.index)
    for action in ["long_straddle", "short_straddle", "long_strangle", "short_strangle"]:
        mask = test["tcn_action"] == action
        realized.loc[mask] = test.loc[mask, f"target_{action}"]
    sharpe = realized.mean() / realized.std()
    rows.append({
        "year": y,
        "n": len(realized),
        "mean_target": realized.mean(),
        "target_sharpe": sharpe,
        "profitable": (realized > 0).mean()
    })
target_sharpe = pd.DataFrame(rows)
print(target_sharpe)
# =============================================================================================================
# =========================== TCN TARGET PERFORMANCE: EXECUTED VS SKIPPED =====================================
# =============================================================================================================

target_cols = [
    "target_long_straddle",
    "target_short_straddle",
    "target_long_strangle",
    "target_short_strangle"
]
rows = []
for y in YEARS:
    test = pd.concat([
        tcn_output_by_year[y][["tcn_action"]],
        tcn_targets_by_year[y][target_cols]
    ], axis=1, sort=False).dropna()
    test = test[test["tcn_action"] != "no_trade"].copy()
    test["target_pnl"] = 0.0
    for action in ["long_straddle", "short_straddle", "long_strangle", "short_strangle"]:
        m = test["tcn_action"] == action
        test.loc[m, "target_pnl"] = test.loc[m, f"target_{action}"]
    entries = tcn_trades[tcn_trades["year"] == y][["entry", "side"]].copy()
    entries["entry"] = pd.to_datetime(entries["entry"], utc=True).dt.floor("h")
    executed = test[test.index.isin(entries["entry"])]
    skipped = test[~test.index.isin(entries["entry"])]
    action_at_entry = test["tcn_action"].reindex(entries["entry"]).to_numpy()
    action_match = (action_at_entry == entries["side"].to_numpy()).mean()
    rows.append({
        "year": y,
        "signals": len(test),
        "executed": len(executed),
        "execution_rate": len(executed) / len(test),
        "action_match": action_match,
        "all_sharpe": test["target_pnl"].mean() / test["target_pnl"].std(),
        "executed_sharpe": executed["target_pnl"].mean() / executed["target_pnl"].std(),
        "skipped_sharpe": skipped["target_pnl"].mean() / skipped["target_pnl"].std()
    })
execution_check = pd.DataFrame(rows)
print(execution_check)


# =========================== REQUIRES STRATEGY_RESULTS =======================================================
# =========================== TCN TARGET PERFORMANCE: EXECUTED VS SKIPPED =====================================
# =============================================================================================================

target_cols = [
    "target_long_straddle",
    "target_short_straddle",
    "target_long_strangle",
    "target_short_strangle"
]

rows = []

for y in YEARS:

    test = pd.concat([
        tcn_output_by_year[y][["tcn_action"]],
        tcn_targets_by_year[y][target_cols]
    ], axis=1, sort=False).dropna()

    test = test[test["tcn_action"] != "no_trade"].copy()

    test["target_pnl"] = 0.0

    for action in ["long_straddle", "short_straddle", "long_strangle", "short_strangle"]:
        m = test["tcn_action"] == action
        test.loc[m, "target_pnl"] = test.loc[m, f"target_{action}"]

    entries = pd.DataFrame(
        strategy_results[f"tcn_mfiv_{str(y)[2:]}"]["greek_tests"]
    )

    entries["time"] = pd.to_datetime(entries["time"], utc=True).dt.floor("h")
    entries = entries.drop_duplicates("time")

    covered = entries[entries["time"].isin(test.index)].copy()

    predicted = test["tcn_action"].reindex(covered["time"]).to_numpy()
    action_match = (predicted == covered["tcn_action"].to_numpy()).mean()

    executed = test[test.index.isin(covered["time"])]
    skipped = test[~test.index.isin(covered["time"])]

    rows.append({
        "year": y,
        "signals": len(test),
        "successful_entries": len(entries),
        "entries_with_target": len(covered),
        "target_coverage": len(covered) / len(entries),
        "action_match": action_match,
        "all_sharpe": test["target_pnl"].mean() / test["target_pnl"].std(),
        "executed_sharpe": executed["target_pnl"].mean() / executed["target_pnl"].std(),
        "skipped_sharpe": skipped["target_pnl"].mean() / skipped["target_pnl"].std()
    })

execution_check = pd.DataFrame(rows)

print(execution_check)
# This is the key result: the TCN itself is selecting the correct action at executed entries (action_match = 1.0),
# but the subset of signals that the sequential trader actually gets to trade is bad in 2019 and 2020.
# my next question : does tcn_best_score actually rank better opportunities?
# =============================================================================================================
# ============================= TCN_SCORE_CALIBRATION_DIAGNOSTIC ========================================
# =============================================================================================================
rows = []
for y in YEARS:
    test = pd.concat([
        tcn_output_by_year[y][["tcn_action", "tcn_best_score"]],
        tcn_targets_by_year[y][target_cols]
    ], axis=1, sort=False).dropna()

    test = test[test["tcn_action"] != "no_trade"].copy()
    test["target_pnl"] = 0.0
    for action in ["long_straddle", "short_straddle", "long_strangle", "short_strangle"]:
        m = test["tcn_action"] == action
        test.loc[m, "target_pnl"] = test.loc[m, f"target_{action}"]
        
    test["score_bin"] = pd.qcut(
        test["tcn_best_score"],
        5,
        labels=False,
        duplicates="drop"
    )

    x = (
        test.groupby("score_bin")
        .agg(
            n=("target_pnl", "size"),
            mean_score=("tcn_best_score", "mean"),
            mean_target=("target_pnl", "mean"),
            sharpe=("target_pnl", lambda z: z.mean() / z.std()),
            profitable=("target_pnl", lambda z: (z > 0).mean())
        )
        .reset_index()
    )

    x["year"] = y
    rows.append(x)

score_check = pd.concat(rows, ignore_index=True)

print(score_check[[
    "year", "score_bin", "n", "mean_score",
    "mean_target", "sharpe", "profitable"
]].to_string(index=False))
# This is the first strong calibration signal. tcn_best_score is clearly informative.

# =============================================================================================================
# ================================= TCN_SCORE_THRESHOLD_CALIBRATION ===========================================
# =============================================================================================================

tests = {}

for y in YEARS:

    x = pd.concat([
        tcn_output_by_year[y][["tcn_action", "tcn_best_score"]],
        tcn_targets_by_year[y][target_cols]
    ], axis=1, sort=False).dropna()

    x = x[x["tcn_action"] != "no_trade"].copy()
    x["target_pnl"] = 0.0

    for action in ["long_straddle", "short_straddle", "long_strangle", "short_strangle"]:
        m = x["tcn_action"] == action
        x.loc[m, "target_pnl"] = x.loc[m, f"target_{action}"]

    tests[y] = x


scores_all = pd.concat([tests[y]["tcn_best_score"] for y in YEARS])

rows = []

for q in [0, .20, .40, .50, .60, .70, .80, .85, .90]:

    threshold = scores_all.quantile(q)

    for y in YEARS:

        x = tests[y][tests[y]["tcn_best_score"] >= threshold]
        sharpe = x["target_pnl"].mean() / x["target_pnl"].std()

        rows.append({
            "quantile": q,
            "threshold": threshold,
            "year": y,
            "n": len(x),
            "mean_target": x["target_pnl"].mean(),
            "sharpe": sharpe
        })

threshold_check = pd.DataFrame(rows)

print(threshold_check.to_string(index=False))

## threshold ≈ 0 → 2033 trades, Sharpe 0.060
## threshold ≈ 0.003 → 508 trades, Sharpe 0.180
## threshold ≈ 0.0074 → 198 trades, Sharpe 0.327
## threshold ≈ 0.0117 → 122 trades, Sharpe 0.374
## TCN SCORE DIAGNOSTIC:
## Check whether larger predicted TCN scores correspond to better realized target PnL.
## Higher score bins = stronger TCN predicted opportunities.
## Result: higher-score trades generally have higher realized Sharpe, especially in 2020 and 2021.
## Therefore tcn_best_score can potentially be used as an entry threshold during calibration.
## It is “being more selective using the TCN's own score improves Sharpe.
## We can be fairly confident that higher tcn_best_score identifies better opportunities inside the fitted target sample. The pattern is strong and appears independently in all three years.
## But we cannot yet say the actual trader Sharpe will improve. The trader only executes a small sequential subset of signals, which was exactly where 2019/2020 deteriorated.

# =============================================================================================================
# ================================ TCN_SCORE_THRESHOLD_TRADER_CALIBRATION =====================================
# =============================================================================================================

sys.modules.pop("srcB.trader_testing_tcn", None)
from srcB.trader_testing_tcn import testing_execute_trading_strategy

THRESHOLDS = [0.0, 0.003, 0.0074, 0.0090, 0.0117]

rows = []

for threshold in THRESHOLDS:
    for y in YEARS:

        market = Market(
            option_df=df_raw_by_year[y],
            time_df=d_btc_raw_by_year[y],
            starting_cash=100_000,
            im_multiplier=0.15,
            mm_buffer_rate=0.05,
        )

        snapshots, greek_tests = testing_execute_trading_strategy(
            bt_year=y,
            bt_quarter=None,
            model="tcn_mfiv",
            outputs_by_year=outputs_by_year,
            market=market,
            trader_cache=trader_cache,
            run_by="year",
            tcn_score_threshold=threshold
        )

        equity = pd.Series(
            [s["equity"] for s in snapshots],
            index=[s["time"] for s in snapshots]
        )

        returns = equity.pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(24 * 365)

        rows.append({
            "threshold": threshold,
            "year": y,
            "trades": len(greek_tests),
            "sharpe": sharpe
        })

threshold_trader_check = pd.DataFrame(rows)

print(threshold_trader_check.to_string(index=False))

# ============================================= POST ESTIMATION ====================================================
# ============================================= PLOTLY SPOT PLOT  ==================================================
# ==================================================================================================================
import pandas as pd
import plotly.graph_objects as go
MODELS_PLOTLY = ["garch", "garch_n_mfiv"]
y = 2020
price_df = d_btc_raw_by_year[y].copy()
price_df = price_df.loc[~price_df.index.duplicated(keep="first")].sort_index()
spot = pd.to_numeric(price_df["close"], errors="coerce")
for MODEL in MODELS_PLOTLY:
    snapshots = strategy_results[f"{MODEL}_{str(y)[2:]}"]["snapshots"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spot.index, y=spot, mode="lines", name="Spot"))

    previous_open = False
    entry_time = None
    entry_positions = []

    for snap in snapshots:
        now = pd.Timestamp(snap["time"]).floor("h")
        positions = snap.get("positions", [])
        currently_open = bool(positions)

        if currently_open and not previous_open:
            entry_time = now
            entry_positions = positions.copy()

        if not currently_open and previous_open:
            exit_time = now

            for position in entry_positions:
                strike = position["strike"]

                fig.add_trace(go.Scatter(
                    x=[entry_time], y=[strike],
                    mode="markers",
                    marker={"size": 6},
                    showlegend=False
                ))

                fig.add_trace(go.Scatter(
                    x=[entry_time, exit_time],
                    y=[strike, strike],
                    mode="lines",
                    line={"dash": "dot"},
                    showlegend=False
                ))
            entry_time = None
            entry_positions = []
        previous_open = currently_open

    fig.update_layout(
        title=f"{MODEL} — {y}",
        xaxis_title="Time",
        yaxis_title="Spot / Strike",
        hovermode="closest"
    )
    fig.update_xaxes(rangeslider_visible=True)
    fig.show()

# ==================================================================================================================
# ======================================= NOTE =====================================================================
# ==================================================================================================================
# below is a copy of output from TCN_SCORE_THRESHOLD_TRADER_CALIBRATION and we have a clear in sample optimal parameter
# The actual sequential trader confirms that score filtering can improve performance,
# but "importantly" the relationship is not monotonic once trader path dependence enters. Earlier it seemed monotonic.
# The most important finding is 0.003. It improves Sharpe versus baseline in every year,
# turns 2020 positive, and still retains 112, 125, and 223 executed trades. 
# That is much more statistically credible than 0.0117, where 2019 and 2020 have only 19 and 17 trades.

# threshold  year  trades    sharpe
#     0.0000  2019     206  0.007511
#     0.0000  2020     238 -0.020190
#     0.0000  2021     250  0.897632
#     0.0030  2019     112  0.390607
#     0.0030  2020     125  0.042897
#     0.0030  2021     223  1.332264
#     0.0074  2019      33  0.257431
#     0.0074  2020      37 -0.342683
#     0.0074  2021     147  1.610950
#     0.0090  2019      26  0.660726
#     0.0090  2020      31 -0.234558
#     0.0090  2021     111  1.461598
#     0.0117  2019      19  0.419985
#     0.0117  2020      17  0.249815
#     0.0117  2021      73  1.472884

# FINAL SELECTION : tcn_score_threshold = 0.003

# ==================================================================================================================
# ======================================= OTHER NOTE ===============================================================
# ==================================================================================================================
# For garch family models : signal is the relative spread between a forecast and trailing rv.
# For regime_switching_mfiv : signal is the spread between MFIV horizon H and rv over that same horizon.
# Entry Logic : fires long when signal is in the top decile of its own rolling distribution. (s_now ≥ q_hi). opposite for short
# So when forecast is unusually far below realized, it goes short vol(strangle).
# That is "trade the direction the model expects variance to move" strategy. buy vol expansion, sell vol contraction.
# Rather than the classic "sell rick implied vol, dont care about direction"

