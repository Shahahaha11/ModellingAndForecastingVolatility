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

# COMES FROM MODELLING.ipynb
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

# ### converts annualized decimal-squared variance into cumulative decimal variance over the next H hours NO MORE USED
# MFIV_HORIZONS = list(range(2, 577))
# QUARTERS = ["q1", "q2", "q3", "q4"]
# for y in YEARS:
#     for q in QUARTERS:
#         out_full = outputs_by_year[y][q].copy()
#         for H in MFIV_HORIZONS:
#             col = f"MFIV_{H}"
#             out_full[col] = ( out_full[col].astype(float) * H / (8760.0) )
#         outputs_by_year[y][q] = out_full


# ==================================================================================================================
# ======================================= LOAD XQ_BY_YEAR USING X_BY_YEAR ==========================================
# ==================================================================================================================
YEARS = [2019, 2020, 2021]

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/x_by_year_all/x_by_year.pkl", "rb") as f:
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

# CHECK MODELLING FILE
outputs_by_year[2021]["q2"].iloc[-1]
outputs_by_year[2021]["q2"].columns

# Equation (outputs_by_year) MFIV_H = (Xq_by_year) MFIV_H × (H / 8760)^2 

# # =============================================================================================================
# # ======================================= SCALING ONLY ON XQ  =================================================
# # ========================== CONVERTS TO CUMULATIVE H-HOUR DECIMAL VARIANCE ====================================

# MFIVS = list(range(2, 577))
# QUARTERS = ["q1", "q2", "q3", "q4"]

# for y in YEARS:
#     for q in QUARTERS:
#         for H in MFIVS:
#             Xq_by_year[y][q][f"MFIV_{H}"] *= (H / 8760.0) ** 2

# =============================================================================================================
# ======================================= SCALING ONLY ON XQ  =================================================
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

# # =============================================================================================================
# # ======================================= ESTIMATION ==========================================================
# # =============================================================================================================
# YEARS = [2019, 2020, 2021]
# QUARTERS = ["q1", "q2", "q3", "q4"]

# THE_COLS = [f"MFIV_{h}" for h in range(2, 577)]

# for y in YEARS:
#     for q in QUARTERS:
#         Xq_df = Xq_by_year[y][q]
#         f_rollB, y_true, idx, regime_state, selected_horizon = regime_switching_mfiv(Xq_df, THE_COLS)

#         idx_by_year[y][q] = idx
#         f_rollB_by_year[y][q] = f_rollB
#         rolling_forecasts[y][q] = f_rollB
#         rolling_residuals[y][q] = y_true - f_rollB
#         y_pred_by_year[y][q] = f_rollB
#         y_true_by_year[y][q] = y_true

#         outputs_by_year[y][q]["regime_switching_mfiv"] = pd.Series(f_rollB, index=Xq_df.index[idx])
#         outputs_by_year[y][q]["regime_state"] = pd.Series(regime_state, index=Xq_df.index[idx] )
#         outputs_by_year[y][q]["selected_horizon"] = pd.Series( selected_horizon, index=Xq_df.index[idx])

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
        # outputs_by_year[y][q]["regime_switching_mfiv"] = f_q
        # outputs_by_year[y][q]["regime_state"] = regime_q
        # outputs_by_year[y][q]["selected_horizon"] = horizon_q

# ==================================================================================================================
# SAVE THIS OUTPUTS
# import pickle
# with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/outputs_by_year_analysis.pkl", "wb") as f:
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

d_btc_raw_by_year[2021]["rfr"] = d_btc_by_year[2021]["rfr"].reindex(d_btc_raw_by_year[2021].index, method="ffill")
d_btc_raw_by_year[2021]["rfr"]

# ==================================================================================================================
# ======================================= SAVE 04  ==========================================================
# ==================================================================================================================
# import pickle
# with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/outputs_by_year/outputs_by_year_04.pkl", "wb") as f:
#     pickle.dump(outputs_by_year, f)

# ===========================================================================================================
# ==================================== OBJECTS YOU NEED BEFORE RUNNING BELOW ================================
# ===========================================================================================================
# outputs_by_year

import pickle 
with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/outputs_by_year/outputs_by_year_04.pkl", "rb") as f:
    outputs_by_year = pickle.load(f)


# # IMPORTS FROM SRC
from srcB.market import *
from srcB.trader_testing import testing_execute_trading_strategy
from srcB.helpers import *

# # READ D_BTC AND DF
# from pathlib import Path
# DATA_DIR = Path("../data/rawData")
# YEARS = [2019, 2020, 2021]
# d_btc_raw_by_year = {}
# df_raw_by_year = {}
# for y in YEARS:
#     d_btc_raw_by_year[y] = pd.read_parquet(DATA_DIR / f"btc_underlying_minutely_{y}.parquet")
#     df_raw_by_year[y]    = pd.read_parquet(DATA_DIR / f"derivative_data_{y}.parquet")

# # SET INDEX TO DATETIME
# for y in YEARS:
#     d_btc_raw_by_year[y] = d_btc_raw_by_year[y].set_index(pd.to_datetime(d_btc_raw_by_year[y]["date_time"]))
#     df_raw_by_year[y] = df_raw_by_year[y].set_index(pd.to_datetime(df_raw_by_year[y]["date_time"]))

outputs_by_year[2021]["q2"].regime_state.nunique()

# ==================================================================================================================
# ======================================= TESTING MAIN  ==========================================================
# ==================================================================================================================
sys.modules.pop("srcB", None)

from srcB.trader_testing import testing_execute_trading_strategy
YEAR = 2021
MODEL = "garch_n_mfiv" 

market = Market(
    option_df=df_raw_by_year[YEAR],
    time_df=d_btc_raw_by_year[YEAR],
    starting_cash=100_000,
    im_multiplier=0.15,
    mm_buffer_rate=0.05,
)

greek_tests_2021 = []

for q in ["q1", "q2", "q3", "q4"]:
    snapshots, greek_tests = testing_execute_trading_strategy(
        bt_year=YEAR, bt_quarter=q, model=MODEL, outputs_by_year=outputs_by_year, market=market, quantity=10, expiry_days=30, hold_days=7, lookback_hours=60, mean_hours=60, ql=0.2, qh=0.70, z=1.645
        )

    greek_tests_2021.extend(greek_tests)

# ==================================================================================================================
print(len(greek_tests_2021))
print(greek_tests_2021[:1])
greek_tests_2021
print(pd.Series([x["action"] for x in market.transactions]).value_counts())
 # ==================================================================================================================
# ============================================= PLOTTING  ==========================================================
# ==================================================================================================================
import matplotlib.pyplot as plt

M = 106
N = 110

for x in greek_tests_2021[M:N]:

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

# ====================================================== ROUGH ============================================================
# ====================================================== ROUGH ============================================================
outputs_by_year[2021]["q2"]
outputs_by_year[y][q]
x = greek_tests_2021[20]

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
import matplotlib.pyplot as plt

plt.plot(
    [x["time"] for x in greek_tests_2021],
    [x["entry_equity"] for x in greek_tests_2021]
)
plt.show()
# ==================================================================================================================
# ======================================= NOTE ==========================================================
# ==================================================================================================================

# For garch family models : signal is the relative spread between a forecast and trailing rv.
# For regime_switching_mfiv : signal is the spread between MFIV horizon H and rv over that same horizon.
# Entry Logic : fires long when signal is in the top decile of its own rolling distribution. (s_now ≥ q_hi). opposite for short
# So when forecast is unusually far below realized, it goes short vol(strangle).
# That is "trade the direction the model expects variance to move" strategy. buy vol expansion, sell vol contraction.
# Rather than the classic "sell rick implied vol, dont care about direction"

