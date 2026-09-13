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
with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/outputs_by_year/outputs_by_year_01.pkl", "rb") as f:
    outputs_by_year = pickle.load(f)


# IMPORTS FROM SRC
from srcB.market import *
from srcB.trader_testing_tcn import testing_execute_trading_strategy
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

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/x_by_year_fut/x_by_year.pkl", "rb") as f:
    X_by_year = pd.read_pickle(f)

YEARS = [2019, 2020, 2021]
QUARTERS = ["q1", "q2", "q3", "q4"]

for y in YEARS:
    X_by_year[y] = X_by_year[y].copy()
    X_by_year[y]["rv"] = X_by_year[y]["rt"] ** 2

Xq_by_year = {}

for y in YEARS:
    Xq_by_year[y] = {}

    Xq_by_year[y]["q1"] = X_by_year[y].loc[f"{y}-01-01" : f"{y}-03-31 23:59:59"].copy()

    Xq_by_year[y]["q2"] = X_by_year[y].loc[f"{y}-04-01" : f"{y}-06-30 23:59:59"].copy()

    Xq_by_year[y]["q3"] = X_by_year[y].loc[f"{y}-07-01" : f"{y}-09-30 23:59:59"].copy()

    Xq_by_year[y]["q4"] = X_by_year[y].loc[f"{y}-10-01" : f"{y}-12-31 23:59:59"].copy()


# ==================================================================================================================
# ======================================= MFIV SCALING =============================================================
# ================== CONVERT ANNUALIZED MFIV -> CUMULATIVE H-HOUR DECIMAL VARIANCE ===============================
# ==================================================================================================================

MFIVS = list(range(2, 577))

for y in YEARS:
    for q in QUARTERS:
        for H in MFIVS:
            Xq_by_year[y][q][f"MFIV_{H}"] *= H / 8760.0


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






# # =============================== LOAD OPTIMAL PARAMETERS IN hyperparam_tuning.py ==================================
with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/hyperparameters/tcn_best_params.pkl", "rb") as f:
    best_tcn_params = pickle.load(f)

# =============================================================================================================
# ======================================= TCN HORIZON TEST =====================================================
# =============================================================================================================

import copy

from srcB.trader_testing_tcn_v1 import testing_execute_trading_strategy
from srcB.tcn_model_mfiv import *

TESTING_HORIZONS = [12, 24, 36, 48, 72, 96]

YEARS = [2019, 2020, 2021]
QUARTERS = ["q1", "q2", "q3", "q4"]

outputs_by_year_base = copy.deepcopy(outputs_by_year)

horizon_results = {}


for H in TESTING_HORIZONS:

    outputs_by_year = copy.deepcopy(outputs_by_year_base)

    # =========================================================================================================
    # ======================================= YEARLY_TCN_ESTIMATION ============================================
    # =========================================================================================================

    tcn_bundle_by_year = {}
    tcn_targets_by_year = {}
    tcn_output_by_year = {}

    for y in YEARS:

        X_df = pd.concat([
            Xq_by_year[y][q]
            for q in QUARTERS
        ]).sort_index().copy()

        garch_df = pd.concat([
            outputs_by_year[y][q]["garch"]
            for q in QUARTERS
        ]).sort_index()

        X_df["garch"] = garch_df.reindex(X_df.index)

        tcn_output, tcn_bundle, tcn_targets = tcn_mfiv(
            X_df=X_df,
            option_df=df_raw_by_year[y],
            time_df=d_btc_raw_by_year[y],
            sequence_length=168,
            forward_hours=H,
            delta_target=0.20,
            channels=best_tcn_params["channels"],
            kernel_size=best_tcn_params["kernel_size"],
            dropout=best_tcn_params["dropout"],
            epochs=best_tcn_params["epochs_by_year"][y],
            batch_size=best_tcn_params["batch_size"],
            learning_rate=best_tcn_params["learning_rate"],
            weight_decay=best_tcn_params["weight_decay"]
        )

        tcn_output_by_year[y] = tcn_output
        tcn_bundle_by_year[y] = tcn_bundle
        tcn_targets_by_year[y] = tcn_targets

        for q in QUARTERS:

            q_index = Xq_by_year[y][q].index
            tcn_q = tcn_output.reindex(q_index)

            outputs_by_year[y][q] = pd.concat([
                outputs_by_year[y][q],
                tcn_q
            ], axis=1)


    # =========================================================================================================
    # ======================================= PREPARE TRADER INPUT =============================================
    # =========================================================================================================

    for y in YEARS:

        for q in QUARTERS:

            outputs_by_year[y][q]["rv"] = Xq_by_year[y][q]["rv"].reindex(
                outputs_by_year[y][q].index
            )

            outputs_by_year[y][q]["MFIV_24"] = Xq_by_year[y][q]["MFIV_24"].reindex(
                outputs_by_year[y][q].index
            )


    # =========================================================================================================
    # ======================================= CACHE FOR WARMUP =================================================
    # =========================================================================================================

    trader_cache = {
        y: pd.concat([
            outputs_by_year[yy][q]
            for yy in YEARS
            if yy <= y
            for q in QUARTERS
        ]).sort_index()
        for y in YEARS
    }


    # =========================================================================================================
    # ======================================= TESTING_MAIN_WITH_CACHE ==========================================
    # =========================================================================================================

    strategy_results = {}

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
            quantity=10,
            expiry_days=30,
            hold_days=7,
            lookback_hours=168,
            mean_hours=12,
            ql=0.10,
            qh=0.90,
            z=1.645,
            forward_hours=H
        )

        strategy_results[y] = {
            "snapshots": snapshots,
            "greek_tests": greek_tests,
            "final_cash": market.cash
        }


    # =========================================================================================================
    # ======================================= SHARPE ===========================================================
    # =========================================================================================================

    rows = []

    for y in YEARS:

        snapshots = strategy_results[y]["snapshots"]

        equity = pd.Series(
            [s["equity"] for s in snapshots],
            index=[s["time"] for s in snapshots]
        )

        returns = equity.pct_change().dropna()

        sharpe = (
            returns.mean()
            / returns.std()
            * np.sqrt(24 * 365)
        )

        rows.append({
            "year": y,
            "sharpe": sharpe,
            "final_cash": strategy_results[y]["final_cash"],
            "trades": len(strategy_results[y]["greek_tests"])
        })

    horizon_results[str(H)] = pd.DataFrame(rows)

    print(f"\nHORIZON = {H} HOURS")
    print(horizon_results[str(H)].to_string(index=False))


# =============================================================================================================
# ======================================= ALL HORIZON RESULTS ==================================================
# =============================================================================================================

for H in TESTING_HORIZONS:

    print(f"\nHORIZON = {H} HOURS")
    print(horizon_results[str(H)].to_string(index=False))

# ========================================= RESULTS FROM FIRST RUN =========================================
# (.venv01) shah@Shahs-MacBook-Air MFIV_calibration % caffeinate -i python tcn_horizon_test_block.py

# HORIZON = 12 HOURS
#  year    sharpe    final_cash  trades
#  2019 -0.150803  97248.678874     264
#  2020 -0.402361  94006.522892     339
#  2021  0.692612 127054.257437     441

# HORIZON = 24 HOURS
#  year    sharpe    final_cash  trades
#  2019 -0.276879  97006.165302     200
#  2020  0.397966 104330.268916     237
#  2021  1.216888 159082.993952     277

# HORIZON = 36 HOURS
#  year   sharpe    final_cash  trades
#  2019 0.491347 104871.947870     163
#  2020 0.313003 103076.082998     183
#  2021 0.516755 113132.919444     189

# HORIZON = 48 HOURS
#  year   sharpe    final_cash  trades
#  2019 0.449021 103149.379261     134
#  2020 1.086318 110028.857052     148
#  2021 0.453551 109849.762008     145

# HORIZON = 72 HOURS
#  year   sharpe    final_cash  trades
#  2019 0.341282 102521.749119      97
#  2020 0.903989 109636.281896      91
#  2021 0.710313 127261.970606      97

# HORIZON = 96 HOURS
#  year   sharpe    final_cash  trades
#  2019 0.691124 106345.857984      73
#  2020 0.889091 109187.438824      78
#  2021 0.508669 113567.214568      79

# (.venv01) shah@Shahs-MacBook-Air MFIV_calibration % 

# ========================================= RESULTS FROM ANOTHER RUN =========================================
