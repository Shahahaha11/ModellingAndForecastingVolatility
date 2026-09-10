# VALIDATION FILE
# SEARCH WORDS :
# FINAL_READY_OUTPUTS_BY_YEAR

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

# ====================================== NOTE : CAUTION! DONT LOOK ITS OOS ================================================
import pickle 
with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/outputs_by_year/outputs_by_year_03.pkl", "rb") as f:
    outputs_by_year = pickle.load(f)
with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/OOS/outputs_by_year/outputs_by_year_03.pkl", "rb") as f:
    outputs_by_year_OOS = pickle.load(f)


# READ D_BTC AND DF
from pathlib import Path
DATA_DIR = Path("../data/rawData")
YEARS = [2019, 2020, 2021, 2022]
d_btc_raw_by_year = {}
df_raw_by_year = {}
for y in YEARS:
    d_btc_raw_by_year[y] = pd.read_parquet(DATA_DIR / f"btc_underlying_minutely_{y}.parquet")
    df_raw_by_year[y]    = pd.read_parquet(DATA_DIR / f"derivative_data_{y}.parquet")

# SET INDEX TO DATETIME
for y in YEARS:
    d_btc_raw_by_year[y] = d_btc_raw_by_year[y].set_index(pd.to_datetime(d_btc_raw_by_year[y]["date_time"]))
    df_raw_by_year[y] = df_raw_by_year[y].set_index(pd.to_datetime(df_raw_by_year[y]["date_time"]))


YEARS = [2019, 2020, 2021, 2022]

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/x_by_year_fut/x_by_year.pkl", "rb") as f:
    X_by_year = pd.read_pickle(f)
with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/OOS/x_by_year_fut/x_by_year.pkl", "rb") as f:
    X_by_year_OOS = pd.read_pickle(f)

# =============================================================================================================
# ======================================= VALIDATION_YEAR =====================================================
# =============================================================================================================

CALIBRATION_YEARS = [2019, 2020, 2021]
VALIDATION_YEAR = 2022
YEARS = CALIBRATION_YEARS + [VALIDATION_YEAR]

X_by_year[VALIDATION_YEAR] = X_by_year_OOS[VALIDATION_YEAR]
outputs_by_year[VALIDATION_YEAR] = outputs_by_year_OOS[VALIDATION_YEAR]
# ======================================= CONTINUE =====================================================

for y in YEARS:
    X_by_year[y] = X_by_year[y].copy()
    X_by_year[y]["rv"] = X_by_year[y]["rt"] ** 2

Xq_by_year = {}
YEARS = [2019, 2020, 2021, 2022]
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


for y in YEARS:
    for q in QUARTERS:
        Xq_by_year[y][q] = Xq_by_year[y][q].loc[
            :, ~Xq_by_year[y][q].columns.duplicated(keep="first")
        ]

# ==================================================================================================================
# ============================================= BORROW RFR COLUMN  =================================================
# ==================================================================================================================
import pickle
with open ("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/ready_data/d_btc_by_year_all.pkl", "rb") as f :
    d_btc_by_year = pickle.load(f)
with open ("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/OOS/ready_data/d_btc_by_year_all.pkl", "rb") as f :
    d_btc_by_year_OOS = pickle.load(f)

d_btc_by_year[VALIDATION_YEAR] = d_btc_by_year_OOS[VALIDATION_YEAR]

YEARS = [2019, 2020, 2021, 2022]
for y in YEARS:
    d_btc_raw_by_year[y]["rfr"] = d_btc_by_year[y]["rfr"].reindex(d_btc_raw_by_year[y].index, method="ffill")

d_btc_raw_by_year[2021]["rfr"]

# =============================================================================================================
# ======================================= CHECKPOINT 1. RUN UNTIL HERE ========================================
# =============================================================================================================

# ==================================================================================================================
# ======================================= LOAD CALIBRATED TCN BUNDLES =============================================
# ==================================================================================================================

sys.modules.pop("srcB.tcn_model_v2", None)
from srcB.tcn_model_mfiv import predict_tcn, build_tcn_features

import pickle
with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/tcn/tcn_bundle_by_year.pkl", "rb") as f:
    tcn_bundle_by_year = pickle.load(f)

print(tcn_bundle_by_year.keys())



# ==================================================================================================================
# ======================================= PREPARE TCN VALIDATION INPUT =============================================
# ==================================================================================================================

QUARTERS = ["q1", "q2", "q3", "q4"]

# Include 2021 only as historical feature warmup for the beginning of 2022.
PREDICTION_YEARS = [2021, VALIDATION_YEAR]

X_validation = pd.concat([
    Xq_by_year[y][q]
    for y in PREDICTION_YEARS
    for q in QUARTERS
]).sort_index().copy()

garch_validation = pd.concat([
    outputs_by_year[y][q]["garch"]
    for y in PREDICTION_YEARS
    for q in QUARTERS
]).sort_index()

X_validation["garch"] = garch_validation.reindex(X_validation.index)

# ==================================================================================================================
# ======================================= TCN_VALIDATION_ENSEMBLE_PREDICTION =======================================
# ==================================================================================================================

CALIBRATION_YEARS = [2019, 2020, 2021]

score_cols = [
    "tcn_long_straddle",
    "tcn_short_straddle",
    "tcn_long_strangle",
    "tcn_short_strangle"
]

tcn_validation_by_model = {}

for y in CALIBRATION_YEARS:
    tcn_validation_by_model[y] = predict_tcn(
        tcn_bundle_by_year[y],
        X_validation
    )

tcn_validation_output = tcn_validation_by_model[2019][score_cols].copy()

for y in [2020, 2021]:
    tcn_validation_output += tcn_validation_by_model[y][score_cols]

tcn_validation_output /= len(CALIBRATION_YEARS)

tcn_validation_output["tcn_best_score"] = (
    tcn_validation_output[score_cols].max(axis=1)
)

best = (
    tcn_validation_output[score_cols]
    .idxmax(axis=1)
    .str.replace("tcn_", "", regex=False)
)

tcn_validation_output["tcn_action"] = np.where(
    tcn_validation_output["tcn_best_score"] > 0,
    best,
    "no_trade"
)

tcn_validation_output = tcn_validation_output[
    tcn_validation_output.index.year == VALIDATION_YEAR
].copy()

## ============================= TEMP CHECK =============================
from srcB.tcn_model_mfiv import predict_tcn, build_tcn_features
print(tcn_validation_output[["tcn_action", "tcn_best_score"]].head())
print(tcn_validation_output["tcn_action"].value_counts())
## ========== are features complete ?
print(X_validation.index.min(), X_validation.index.max())
features_check = build_tcn_features(X_validation)
print(features_check.loc["2021-12-25":"2022-01-10"].isna().sum())
print(features_check.loc["2021-12-25":"2022-01-10"].dropna().index.min())



# ==================================================================================================================
# ======================================= ADD TCN VALIDATION OUTPUT =================================================
# ==================================================================================================================

for q in QUARTERS:
    q_index = Xq_by_year[VALIDATION_YEAR][q].index
    tcn_q = tcn_validation_output.reindex(q_index)

    outputs_by_year[VALIDATION_YEAR][q] = outputs_by_year[VALIDATION_YEAR][q].drop(
        columns=tcn_q.columns,
        errors="ignore"
    )

    outputs_by_year[VALIDATION_YEAR][q] = pd.concat(
        [outputs_by_year[VALIDATION_YEAR][q], tcn_q],
        axis=1
    )

print(
    outputs_by_year[VALIDATION_YEAR]["q1"][
        ["tcn_action", "tcn_best_score"]
    ].dropna().head()
)


# ======================================= REQUIRED DUE TO PREVIOUS SAVE ==============================================
# for q in QUARTERS:
#     outputs_by_year[VALIDATION_YEAR][q] = outputs_by_year[VALIDATION_YEAR][q].loc[
#         :, ~outputs_by_year[VALIDATION_YEAR][q].columns.duplicated(keep="last")
#     ]

# print(outputs_by_year[2022]["q1"][["tcn_action", "tcn_best_score"]].dropna().head())

# ==================================================================================================================
# ======================================= PREPARE VALIDATION TRADER INPUT ==========================================
# ==================================================================================================================

for q in QUARTERS:
    outputs_by_year[VALIDATION_YEAR][q]["rv"] = Xq_by_year[VALIDATION_YEAR][q]["rv"].reindex(
        outputs_by_year[VALIDATION_YEAR][q].index
    )

    outputs_by_year[VALIDATION_YEAR][q]["MFIV_24"] = Xq_by_year[VALIDATION_YEAR][q]["MFIV_24"].reindex(
        outputs_by_year[VALIDATION_YEAR][q].index
    )

trader_cache = {
    VALIDATION_YEAR: pd.concat(
        [outputs_by_year[VALIDATION_YEAR][q] for q in QUARTERS]
    ).sort_index()
}
# ==================================================================================================================
# ======================================= TCN 2022 VALIDATION =======================================================
# ==================================================================================================================

from srcB.market import Market

sys.modules.pop("srcB.trader_testing_tcn", None)
from srcB.trader_testing_tcn import testing_execute_trading_strategy

market = Market(
    option_df=df_raw_by_year[VALIDATION_YEAR],
    time_df=d_btc_raw_by_year[VALIDATION_YEAR],
    starting_cash=100_000,
    im_multiplier=0.15,
    mm_buffer_rate=0.05,
)

snapshots, greek_tests = testing_execute_trading_strategy(
    bt_year=VALIDATION_YEAR,
    bt_quarter=None,
    model="tcn_mfiv",
    outputs_by_year=outputs_by_year,
    market=market,
    trader_cache=trader_cache,
    run_by="year",
    tcn_score_threshold=0.003
)

equity = pd.Series(
    [s["equity"] for s in snapshots],
    index=[s["time"] for s in snapshots]
)

returns = equity.pct_change().dropna()
sharpe = returns.mean() / returns.std() * np.sqrt(24 * 365)

print("year:", VALIDATION_YEAR)
print("trades:", len(greek_tests))
print("sharpe:", sharpe)
print("final equity:", equity.iloc[-1])


rfr_annual = 0.045
rfr_hourly = (1 + rfr_annual) ** (1 / (24 * 365)) - 1

excess_returns = returns - rfr_hourly

sharpe = (excess_returns.mean() / excess_returns.std()* np.sqrt(24 * 365))

sharpe