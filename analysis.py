# Further files with the same prefix means progress. This ones has less extra cells. Only what is needed.
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

### NOTE FIRST IN THE MORNING : THIS USES OLD MFIVS, SO ONLY 04 I WILL SAVE NOW IS THE TRUE MODEL
### I HAVE TO REESTIMATE MODELLING.IPYNB
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

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/x_by_year_all_interpol2/x_by_year.pkl", "rb") as f:
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

# ==================================================================================================================
# ======================================= CHECK  ======================================================
# contains raw horizon-dependent MFIV
Xq_by_year[2021]["q2"].iloc[-1]
Xq_by_year[2021]["q2"].columns

# # CHECK MODELLING FILE
# outputs_by_year[2021]["q2"].iloc[-1]
# outputs_by_year[2021]["q2"].columns

# =============================================================================================================
# ============== MFIV SCALING ============= SCALING ONLY ON XQ  =================================================
# ================== CONVERTS ANNUALIZED MFIV -> CUMULATIVE H-HOUR DECIMAL VARIANCE ===========================

MFIVS = list(range(2, 577))
QUARTERS = ["q1", "q2", "q3", "q4"]

for y in YEARS:
    for q in QUARTERS:
        for H in MFIVS:
            Xq_by_year[y][q][f"MFIV_{H}"] *= (H / 8760.0)

# =============================================================================================================
# ======================================= CHECKPOINT 1. RUN UNTIL HERE =====================================================================================
# =============================================================================================================

# =============================================================================================================
# ======================================= PREPARE EMPTY DICTS =================================================
# =============================================================================================================

YEARS = [2019, 2020, 2021]
QUARTERS = ["q1", "q2", "q3", "q4"]

idx_by_year = {y : {} for y in YEARS}
f_rollB_by_year = {y: {} for y in YEARS}
rolling_forecasts =  {y: {} for y in YEARS}
rolling_residuals = {y: {} for y in YEARS}
y_pred_by_year = {y: {} for y in YEARS}
y_true_by_year = {y: {} for y in YEARS}

# ==================================================================================================================
# ======================================= YEARLY ESTIMATION  ==========================================================
# ==================================================================================================================
# YEARLY ESTIMATION
YEARS = [2019, 2020, 2021]
QUARTERS = ["q1", "q2", "q3", "q4"]

THE_COLS = [f"MFIV_{h}" for h in range(2, 577)]

for y in YEARS:
    X_df = pd.concat([Xq_by_year[y][q] for q in QUARTERS]).sort_index()
    f_rollB, y_true, idx, regime_state, selected_horizon = regime_switching_mfiv( X_df, THE_COLS )
    
    full_forecast = pd.Series(f_rollB, index=X_df.index[idx])
    full_true = pd.Series(y_true, index=X_df.index[idx])
    full_regime = pd.Series(regime_state, index=X_df.index[idx])
    full_horizon = pd.Series(selected_horizon, index=X_df.index[idx])

    for q in QUARTERS:
        q_index = Xq_by_year[y][q].index
        f_q = full_forecast.reindex(q_index).dropna()
        y_q = full_true.reindex(f_q.index)
        regime_q = full_regime.reindex(f_q.index)
        horizon_q = full_horizon.reindex(f_q.index)

        idx_q = Xq_by_year[y][q].index.get_indexer(f_q.index)
        idx_by_year[y][q] = idx_q
        f_rollB_by_year[y][q] = f_q.to_numpy()
        rolling_forecasts[y][q] = f_q.to_numpy()
        rolling_residuals[y][q] = (y_q - f_q).to_numpy()
        y_pred_by_year[y][q] = f_q.to_numpy()
        y_true_by_year[y][q] = y_q.to_numpy()

        outputs_by_year[y][q] = pd.concat([outputs_by_year[y][q], pd.DataFrame({"regime_switching_mfiv": f_q, "regime_state": regime_q, "selected_horizon" : horizon_q})], axis = 1)
# ==================================================================================================================
# SAVE THIS OUTPUTS
# import pickle
# with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/outputs_by_year_analysis_interpol2.pkl", "wb") as f:
#     pickle.dump(outputs_by_year, f)
# ==================================================================================================================

# =============================================================================================================
# ======================================= CHECKPOINT 2. RUN FROM HERE ==================================================================================================
# =============================================================================================================

# ==================================================================================================================
# ============================================= BORROW RFR COLUMN  =================================================
# ==================================================================================================================
import pickle
with open ("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/ready_data/d_btc_by_year.pkl", "rb") as f :
    d_btc_by_year = pickle.load(f)

YEARS = [2019, 2020, 2021]
for y in YEARS:
    d_btc_raw_by_year[y]["rfr"] = d_btc_by_year[y]["rfr"].reindex(d_btc_raw_by_year[y].index, method="ffill")

d_btc_raw_by_year[2021]["rfr"]

# ==================================================================================================================
# ======================================= SAVE 04 REPEATED ==========================================================
# ==================================================================================================================
# import pickle
# with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/outputs_by_year/outputs_by_year_04_interpol2.pkl", "wb") as f:
#     pickle.dump(outputs_by_year, f)

# ===========================================================================================================
# ==================================== OBJECTS YOU NEED BEFORE RUNNING BELOW ================================
# ===========================================================================================================

import pickle 
with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/outputs_by_year/outputs_by_year_04_interpol2.pkl", "rb") as f:
    outputs_by_year = pickle.load(f)

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
# ======================================= TESTING MAIN WITH CACHE  ==========================================================
# ==================================================================================================================
# sys.modules.pop("srcB", None)
from srcB.trader_testing import testing_execute_trading_strategy

YEARS = [2019, 2020, 2021]
MODELS = ["garch_n_mfiv", "garch"]

RUN_BY = "year"    # "year" or "quarter"

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
                lookback_hours=60,
                mean_hours=60,
                ql=0.2,
                qh=0.70,
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

        globals()[history_key] = equity_curve
# ==================================================================================================================
# ======================================= TESTING MAIN  ==========================================================
# ==================================================================================================================
# from srcB.trader_testing import testing_execute_trading_strategy
# YEARS = [2019, 2020, 2021]
# MODELS = ["garch_n_mfiv" , "garch"]

# strategy_results = {}

# for YEAR in YEARS:
#     for MODEL in MODELS :

#         market = Market(
#             option_df=df_raw_by_year[YEAR],
#             time_df=d_btc_raw_by_year[YEAR],
#             starting_cash=100_000,
#             im_multiplier=0.15,
#             mm_buffer_rate=0.05,
#         )

#         greek_tests = []
#         yearly_snapshots = []

#         for q in ["q1", "q2", "q3", "q4"]:
#             snapshots, greek_tests_q = testing_execute_trading_strategy(
#                 bt_year=YEAR, bt_quarter=q, model=MODEL, outputs_by_year=outputs_by_year, market=market, quantity=10, expiry_days=30, hold_days=7, lookback_hours=60, mean_hours=60, ql=0.2, qh=0.70, z=1.645
#             )

#             greek_tests.extend(greek_tests_q)
#             yearly_snapshots.extend(snapshots)

#         history_key = f"{MODEL}_{str(YEAR)[2:]}"
#         equity_curve = market.get_equity_history()

#         strategy_results[history_key] = {
#             "equity_curve": equity_curve,
#             "snapshots": yearly_snapshots,
#             "greek_tests" : greek_tests,
#             "orders": list(market.orders),
#             "transactions": list(market.transactions),
#             "final_cash": market.cash,
#             "required_margin": market.get_total_required_margin(),
#             "free_liquidity": market.get_free_liquidity(),
#         }

#         globals()[history_key] = equity_curve
# ==================================================================================================================
print(len(greek_tests))
print(greek_tests[:1])
print(pd.Series([x["action"] for x in market.transactions]).value_counts())

# ==================================================================================================================
# ============================================= PLOT EQUITY  ========================================================
# ==================================================================================================================
import matplotlib.pyplot as plt
PLOT_YEAR = 2020
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

# ==================================================================================================================
# ============================================= PLOTLY SPOT PLOT  ==================================================
# ==================================================================================================================
# import pandas as pd
# import plotly.graph_objects as go

# y = 2021
# price_df = d_btc_raw_by_year[y].copy()
# price_df = price_df.loc[~price_df.index.duplicated(keep="first")].sort_index()
# spot = pd.to_numeric(price_df["close"], errors="coerce")
# for MODEL in MODELS:
#     snapshots = strategy_results[f"{MODEL}_{str(y)[2:]}"]["snapshots"]
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=spot.index, y=spot, mode="lines", name="Spot"))

#     previous_open = False
#     entry_time = None
#     entry_positions = []

#     for snap in snapshots:
#         now = pd.Timestamp(snap["time"]).floor("h")
#         positions = snap.get("positions", [])
#         currently_open = bool(positions)

#         if currently_open and not previous_open:
#             entry_time = now
#             entry_positions = positions.copy()

#         if not currently_open and previous_open:
#             exit_time = now

#             for position in entry_positions:
#                 strike = position["strike"]

#                 fig.add_trace(go.Scatter(
#                     x=[entry_time], y=[strike],
#                     mode="markers",
#                     marker={"size": 6},
#                     showlegend=False
#                 ))

#                 fig.add_trace(go.Scatter(
#                     x=[entry_time, exit_time],
#                     y=[strike, strike],
#                     mode="lines",
#                     line={"dash": "dot"},
#                     showlegend=False
#                 ))
#             entry_time = None
#             entry_positions = []
#         previous_open = currently_open

#     fig.update_layout(
#         title=f"{MODEL} — {y}",
#         xaxis_title="Time",
#         yaxis_title="Spot / Strike",
#         hovermode="closest"
#     )
#     fig.update_xaxes(rangeslider_visible=True)
#     fig.show()

# ====================================================== ROUGH ============================================================
x = strategy_results["garch_n_mfiv_21"]["transactions"]

print(x[0])
print(x[-1])
# ========================================================================================================================================
# ====================================================== STRATEGY DIAGNOSTICS ============================================================
# ========================================================================================================================================
import pandas as pd
y = 2020
the_model = "garch_n_mfiv"
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

# ====================================================== ROUGH ============================================================
outputs_by_year[2021]["q2"].regime_state.nunique()
outputs_by_year[2021]["q2"]
outputs_by_year[y][q]

x = pd.concat([outputs_by_year[2021][q] for q in ["q1","q2","q3","q4"]]).sort_index()
s = (x["garch"] - x["rv"]) / x["rv"]
print(x[["garch","rv"]].describe())
print(s.describe())
print("corr:", x["garch"].corr(x["rv"]))
print("garch > rv:", (x["garch"] > x["rv"]).mean())

lo = s.rolling(60).quantile(.2)
hi = s.rolling(60).quantile(.7)
m = (s.index.hour >= 11) & (s.index.hour < 15)
print("long:", (s[m] <= lo[m]).sum())
print("short:", (s[m] >= hi[m]).sum())

# print("time:", x["time"])
# print("spot:", x["spot"])
# print("rfr:", x["rfr"])
# print("call_ivs:", x["call_ivs"])
# print("put_ivs:", x["put_ivs"])

# print("\nCALLS")
# for i in x["selected_calls"]:
#     print(i.Strike, i.Expiry, i.Right, i.Direction)

# print("\nPUTS")
# for i in x["selected_puts"]:
#     print(i.Strike, i.Expiry, i.Right, i.Direction)

# ====================================================== FOR GPT ============================================================
print(df_raw_by_year[2021].dropna(how="all").head(3))
print(d_btc_raw_by_year[2021].dropna(how="all").head(3))
print(outputs_by_year[2021]["q2"].dropna(how="all").head(3))
print(df_raw_by_year[2021].dropna(subset=["iv"]).head(3))
# ==================================================================================================================
# ======================================= NOTE ==========================================================
# ==================================================================================================================

# ==================================================================================================================
# ======================================= NOTE ==========================================================
# ==================================================================================================================

# For garch family models : signal is the relative spread between a forecast and trailing rv.
# For regime_switching_mfiv : signal is the spread between MFIV horizon H and rv over that same horizon.
# Entry Logic : fires long when signal is in the top decile of its own rolling distribution. (s_now ≥ q_hi). opposite for short
# So when forecast is unusually far below realized, it goes short vol(strangle).
# That is "trade the direction the model expects variance to move" strategy. buy vol expansion, sell vol contraction.
# Rather than the classic "sell rick implied vol, dont care about direction"

