# VALIDATION GLOBAL
# SEARCH WORDS :
# GLOBAL_VALIDATION_MODELS
# GLOBAL_VALIDATION_PREDICTIONS
# GLOBAL_VALIDATION_TRADER
# GLOBAL_VALIDATION_RESULTS

import pandas as pd
import numpy as np
from math import sqrt, pi, exp
from arch import arch_model
import yfinance as yf 
from scipy.optimize import minimize
import warnings
from arch.__future__ import reindexing
from arch.utility.exceptions import ConvergenceWarning
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


# =============================================================================================================
# ======================================= LOAD BASE OUTPUTS ===================================================
# =============================================================================================================

import pickle 

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/outputs_by_year/outputs_by_year_04.pkl", "rb") as f:
    outputs_by_year = pickle.load(f)

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/OOS/outputs_by_year/outputs_by_year_01.pkl", "rb") as f:
    outputs_by_year_OOS = pickle.load(f)


# =============================================================================================================
# ======================================= READ D_BTC AND DF ===================================================
# =============================================================================================================

from pathlib import Path

DATA_DIR = Path("../data/rawData")

YEARS = [2019, 2020, 2021, 2022]

d_btc_raw_by_year = {}
df_raw_by_year = {}

for y in YEARS:
    d_btc_raw_by_year[y] = pd.read_parquet(DATA_DIR / f"btc_underlying_minutely_{y}.parquet")
    df_raw_by_year[y] = pd.read_parquet(DATA_DIR / f"derivative_data_{y}.parquet")


# =============================================================================================================
# ======================================= SET INDEX TO DATETIME ===============================================
# =============================================================================================================

for y in YEARS:
    d_btc_raw_by_year[y] = d_btc_raw_by_year[y].set_index(pd.to_datetime(d_btc_raw_by_year[y]["date_time"]))
    df_raw_by_year[y] = df_raw_by_year[y].set_index(pd.to_datetime(df_raw_by_year[y]["date_time"]))
    

# =============================================================================================================
# ======================================= VALIDATION YEAR =====================================================
# =============================================================================================================

CALIBRATION_YEARS = [2019, 2020, 2021]
VALIDATION_YEAR = 2022
YEARS = CALIBRATION_YEARS + [VALIDATION_YEAR]


with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/x_by_year_fut/x_by_year.pkl", "rb") as f:
    X_by_year = pd.read_pickle(f)

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/OOS/x_by_year_fut/x_by_year.pkl", "rb") as f:
    X_by_year_OOS = pd.read_pickle(f)


X_by_year[VALIDATION_YEAR] = X_by_year_OOS[VALIDATION_YEAR]
outputs_by_year[VALIDATION_YEAR] = outputs_by_year_OOS[VALIDATION_YEAR]


for y in YEARS:
    X_by_year[y] = X_by_year[y].copy()
    X_by_year[y]["rv"] = X_by_year[y]["rt"] ** 2


Xq_by_year = {}
QUARTERS = ["q1", "q2", "q3", "q4"]

for y in YEARS:
    Xq_by_year[y] = {}
    Xq_by_year[y]["q1"] = X_by_year[y].loc[f"{y}-01-01" : f"{y}-03-31 23:59:59"]
    Xq_by_year[y]["q2"] = X_by_year[y].loc[f"{y}-04-01" : f"{y}-06-30 23:59:59"]
    Xq_by_year[y]["q3"] = X_by_year[y].loc[f"{y}-07-01" : f"{y}-09-30 23:59:59"]
    Xq_by_year[y]["q4"] = X_by_year[y].loc[f"{y}-10-01" : f"{y}-12-31 23:59:59"]
    
    

# =============================================================================================================
# ======================================= MFIV SCALING ========================================================
# =============================================================================================================
# CONVERTS ANNUALIZED MFIV -> CUMULATIVE H-HOUR DECIMAL VARIANCE

MFIVS = list(range(2, 577))

for y in YEARS:
    for q in QUARTERS:
        for H in MFIVS:
            Xq_by_year[y][q][f"MFIV_{H}"] *= (H / 8760.0)


for y in YEARS:
    for q in QUARTERS:
        Xq_by_year[y][q] = Xq_by_year[y][q].loc[
            :, ~Xq_by_year[y][q].columns.duplicated(keep="first")
        ]
        
        
# =============================================================================================================
# ======================================= BORROW RFR COLUMN ===================================================
# =============================================================================================================

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/ready_data/d_btc_by_year_all.pkl", "rb") as f:
    d_btc_by_year = pickle.load(f)

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/OOS/ready_data/d_btc_by_year_all.pkl", "rb") as f:
    d_btc_by_year_OOS = pickle.load(f)


d_btc_by_year[VALIDATION_YEAR] = d_btc_by_year_OOS[VALIDATION_YEAR]


for y in YEARS:
    d_btc_raw_by_year[y]["rfr"] = d_btc_by_year[y]["rfr"].reindex(
        d_btc_raw_by_year[y].index,
        method="ffill"
    )
    
    

# =============================================================================================================
# ======================================= CHECKPOINT 1 ========================================================
# =============================================================================================================
# =============================================================================================================
# ======================================= FROZEN MODEL SPECIFICATIONS =========================================
# =============================================================================================================

TCN_SCORE_THRESHOLD = 0.003
GRU_SCORE_THRESHOLD = 0.0000
TSM_SCORE_THRESHOLD = 0.0030653050635010004
DL_SCORE_THRESHOLD = 0.005388132412917912
ITR_SCORE_THRESHOLD = 0.0034569846466183662

# =============================================================================================================
# ======================================= LOAD FROZEN MODEL BUNDLES ===========================================
# =============================================================================================================

MODEL_YEAR = 2021

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/TCN/tcn_bundle_by_year.pkl", "rb") as f:
    tcn_bundle_by_year = pickle.load(f)

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/gru/gru_bundle_by_year.pkl", "rb") as f:
    gru_bundle_by_year = pickle.load(f)

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/tsm/tsm_bundle_by_year.pkl", "rb") as f:
    tsm_bundle_by_year = pickle.load(f)

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/dl/dl_bundle_by_year.pkl", "rb") as f:
    dl_bundle_by_year = pickle.load(f)

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/itr/itr_bundle_by_year.pkl", "rb") as f:
    itr_bundle_by_year = pickle.load(f)


tcn_bundle = tcn_bundle_by_year[MODEL_YEAR]
gru_bundle = gru_bundle_by_year[MODEL_YEAR]
tsm_bundle = tsm_bundle_by_year[MODEL_YEAR]
dl_bundle = dl_bundle_by_year[MODEL_YEAR]
itr_bundle = itr_bundle_by_year[MODEL_YEAR]


# =============================================================================================================
# ======================================= IMPORT MODEL PREDICTORS =============================================
# =============================================================================================================

sys.modules.pop("srcB.tcn_model_v2", None)
sys.modules.pop("srcB.gru_model", None)
sys.modules.pop("srcB.tsm_model", None)
sys.modules.pop("srcB.dl_model", None)
sys.modules.pop("srcB.itr_model", None)

from srcB.tcn_model_v2 import predict_tcn
from srcB.gru_model import predict_gru
from srcB.tsm_model import predict_tsm
from srcB.dl_model import predict_dl
from srcB.itr_model import predict_itr

# =============================================================================================================
# ======================================= CHECKPOINT 2 ========================================================
# =============================================================================================================
# OLD:
# 2021 data -> refit five models -> 2022 prediction
# NEW:
# saved HPO-fitted 2021 bundles -> 2022 prediction
# =============================================================================================================
# ======================================= PREPARE VALIDATION INPUT ============================================
# =============================================================================================================

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


# =============================================================================================================
# ======================================= GLOBAL VALIDATION PREDICTIONS =======================================
# =============================================================================================================

tcn_validation_output = predict_tcn(tcn_bundle, X_validation)
gru_validation_output = predict_gru(gru_bundle, X_validation)
tsm_validation_output = predict_tsm(tsm_bundle, X_validation)
dl_validation_output = predict_dl(dl_bundle, X_validation)
itr_validation_output = predict_itr(itr_bundle, X_validation)


tcn_validation_output = tcn_validation_output[
    tcn_validation_output.index.year == VALIDATION_YEAR
].copy()

gru_validation_output = gru_validation_output[
    gru_validation_output.index.year == VALIDATION_YEAR
].copy()

tsm_validation_output = tsm_validation_output[
    tsm_validation_output.index.year == VALIDATION_YEAR
].copy()

dl_validation_output = dl_validation_output[
    dl_validation_output.index.year == VALIDATION_YEAR
].copy()

itr_validation_output = itr_validation_output[
    itr_validation_output.index.year == VALIDATION_YEAR
].copy()


print("TCN:", len(tcn_validation_output))
print("GRU:", len(gru_validation_output))
print("TSM:", len(tsm_validation_output))
print("DL:", len(dl_validation_output))
print("ITR:", len(itr_validation_output))


# =============================================================================================================
# ======================================= ADD VALIDATION OUTPUTS ==============================================
# =============================================================================================================

validation_outputs = [
    tcn_validation_output,
    gru_validation_output,
    tsm_validation_output,
    dl_validation_output,
    itr_validation_output
]

for q in QUARTERS:
    q_index = Xq_by_year[VALIDATION_YEAR][q].index

    for model_output in validation_outputs:
        model_q = model_output.reindex(q_index)

        outputs_by_year[VALIDATION_YEAR][q] = outputs_by_year[VALIDATION_YEAR][q].drop(
            columns=model_q.columns,
            errors="ignore"
        )

        outputs_by_year[VALIDATION_YEAR][q] = pd.concat(
            [outputs_by_year[VALIDATION_YEAR][q], model_q],
            axis=1
        )


# =============================================================================================================
# ======================================= PREPARE VALIDATION TRADER INPUT =====================================
# =============================================================================================================

for q in QUARTERS:
    outputs_by_year[VALIDATION_YEAR][q]["rv"] = Xq_by_year[VALIDATION_YEAR][q]["rv"].reindex(
        outputs_by_year[VALIDATION_YEAR][q].index
    )

    outputs_by_year[VALIDATION_YEAR][q]["MFIV_24"] = Xq_by_year[VALIDATION_YEAR][q]["MFIV_24"].reindex(
        outputs_by_year[VALIDATION_YEAR][q].index
    )
    

# =============================================================================================================
# ======================================= CHECKPOINT 3 ========================================================
# =============================================================================================================
# WE NOW HAVE 2022 OOS PREDICTIONS FROM :
# TCN
# GRU
# TSM
# DLINEAR
# ITRANSFORMER


# The bundle contains the fitted forecasting model and things needed to reproduce its predictions, such as:
# trained PyTorch parameters,
# feature scaler,
# target scaler,
# feature columns,
# sequence length,
# architecture/fitted metadata.
# So from a bundle plus the same feature data, you can reproduce the model's scores/actions.

# To reproduce the exact equity line you additionally need the same:
# 2022 model predictions
# + chosen score threshold
# + option data
# + BTC/time data
# + trader logic
# + Market settings
# + execution/liquidation/fees
# + starting cash



# =============================================================================================================
# ======================================= GLOBAL VALIDATION TRADER ============================================
# =============================================================================================================

from srcB.market import Market

sys.modules.pop("srcB.trader_testing_tcn", None)
from srcB.trader_testing_tcn import testing_execute_trading_strategy


MODEL_CONFIGS = {
    "tcn_mfiv": {
        "action_col": "tcn_action",
        "score_col": "tcn_best_score",
        "threshold": TCN_SCORE_THRESHOLD
    },
    "gru_mfiv": {
        "action_col": "gru_action",
        "score_col": "gru_best_score",
        "threshold": GRU_SCORE_THRESHOLD
    },
    "tsm_mfiv": {
        "action_col": "tsm_action",
        "score_col": "tsm_best_score",
        "threshold": TSM_SCORE_THRESHOLD
    },
    "dl_mfiv": {
        "action_col": "dl_action",
        "score_col": "dl_best_score",
        "threshold": DL_SCORE_THRESHOLD
    },
    "itr_mfiv": {
        "action_col": "itr_action",
        "score_col": "itr_best_score",
        "threshold": ITR_SCORE_THRESHOLD
    }
}


strategy_results = {}

for MODEL, config in MODEL_CONFIGS.items():

    for q in QUARTERS:
        outputs_by_year[VALIDATION_YEAR][q]["tcn_action"] = outputs_by_year[VALIDATION_YEAR][q][
            config["action_col"]
        ]

        outputs_by_year[VALIDATION_YEAR][q]["tcn_best_score"] = outputs_by_year[VALIDATION_YEAR][q][
            config["score_col"]
        ]


    trader_cache = {
        VALIDATION_YEAR: pd.concat(
            [outputs_by_year[VALIDATION_YEAR][q] for q in QUARTERS]
        ).sort_index()
    }


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
        tcn_score_threshold=config["threshold"]
    )


    history_key = f"{MODEL}_{str(VALIDATION_YEAR)[2:]}"

    equity_curve = market.get_equity_history()

    strategy_results[history_key] = {
        "equity_curve": equity_curve,
        "snapshots": snapshots,
        "greek_tests": greek_tests,
        "orders": list(market.orders),
        "transactions": list(market.transactions),
        "final_cash": market.cash,
        "required_margin": market.get_total_required_margin(),
        "free_liquidity": market.get_free_liquidity(),
    }


print(strategy_results.keys())

# The adapter is recreated before each model run, and trader_cache is rebuilt after that overwrite. That is important: otherwise the cache could retain the previous model's copied tcn_action / tcn_best_score columns.
# The native gru_*, tsm_*, dl_*, itr_* columns are never altered. Only the two compatibility columns expected by trader_testing_tcn are overwritten for that particular run.

# =============================================================================================================
# ======================================= VALIDATION SHARPE CHECK =============================================
# =============================================================================================================

MODELS = ["tcn_mfiv", "gru_mfiv", "tsm_mfiv", "dl_mfiv", "itr_mfiv"]
rows = []
for MODEL in MODELS:
    history_key = f"{MODEL}_{str(VALIDATION_YEAR)[2:]}"
    snapshots = strategy_results[history_key]["snapshots"]
    equity = pd.Series(
        [s["equity"] for s in snapshots],
        index=[s["time"] for s in snapshots]
    )
    returns = equity.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(24 * 365)
    rows.append({
        "year": VALIDATION_YEAR,
        "model": MODEL,
        "trades": len(strategy_results[history_key]["greek_tests"]),
        "sharpe": sharpe,
        "final_equity": equity.iloc[-1]
    })
validation_sharpe = pd.DataFrame(rows)
print(validation_sharpe)



# =============================================================================================================
# ======================================= SAVE VALIDATION STRATEGY RESULTS ====================================
# =============================================================================================================

os.makedirs("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/VAL", exist_ok=True)

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/VAL/strategy_results_2022.pkl","wb") as f:
    pickle.dump(strategy_results, f)

print(strategy_results.keys())

# TCN_SCORE_THRESHOLD = 0.003
# Trade only when the best of the four predicted option strategies
# has expected PnL greater than or equal to 0.3% of BTC spot at entry.
