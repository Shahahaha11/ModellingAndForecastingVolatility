# STATUS : RUNNING
# CURRENTLY USES OPTION 2 VALIDATION STYLE. OPTION 4 IS NOT YET KNOWN.
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
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="y is poorly scaled")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import os, sys
sys.path.insert(0, "/Users/shah/CODE_BOOK_4/THESIS_WORKING/THESIS_2")
sys.path.insert(0, "/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/MFIV_calibration")
os.chdir("/Users/shah/CODE_BOOK_4/THESIS_WORKING/THESIS_2")
sys.modules.pop("srcB", None)

import pyttsx3
def speak_done():
    engine = pyttsx3.init()
    engine.say("CELL COMPLETED CELL COMPLETED ")
    engine.runAndWait()
    engine.stop()


# =============================================================================================================
# ======================================= LOAD BASE OUTPUTS ===================================================
# =============================================================================================================

import pickle 
with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/OOS/outputs_by_year/outputs_by_year_01.pkl", "rb") as f:
    outputs_by_year_OOS = pickle.load(f)

# =============================================================================================================
# ======================================= READ D_BTC AND DF ===================================================
# =============================================================================================================

from pathlib import Path

DATA_DIR = Path("../data/rawData")

YEARS = [2022, 2023, 2024, 2025]

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

# ======================================= OOS FRESH X_BY_YEAR =================================================

X_by_year = {}

# =============================================================================================================
# ======================================= OOS YEAR #1 ==========================================================
# =============================================================================================================

FITTING_YEARS = [2022, 2023, 2024]
PREDICT_YEARS = [2023, 2024, 2025]
YEARS = [2022, 2023, 2024, 2025]
QUARTERS = ["q1", "q2", "q3", "q4"]


with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/OOS/x_by_year_fut/x_by_year.pkl", "rb") as f:
    X_by_year_OOS = pickle.load(f)

for y in YEARS:
    X_by_year[y] = X_by_year_OOS[y]

for y in YEARS:
    for q in QUARTERS:
        outputs_by_year_OOS[y][q] = outputs_by_year_OOS[y][q][["garch"]].copy()
# =============================================================================================================

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


# =============================================================================================================
# ======================================= MFIV SCALING ========================================================
# ================== CONVERT ANNUALIZED MFIV -> CUMULATIVE H-HOUR DECIMAL VARIANCE ===========================
# =============================================================================================================

MFIVS = list(range(2, 577))

for y in YEARS:
    for q in QUARTERS:
        for H in MFIVS:
            Xq_by_year[y][q][f"MFIV_{H}"] *= H / 8760.0


# =============================================================================================================
# ======================================= BORROW RFR COLUMN ===================================================
# =============================================================================================================

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/OOS/ready_data/d_btc_by_year_all.pkl", "rb") as f:
    d_btc_by_year_OOS = pickle.load(f)


for y in YEARS:
    d_btc_raw_by_year[y]["rfr"] = d_btc_by_year_OOS[y]["rfr"].reindex(
        d_btc_raw_by_year[y].index,
        method="ffill"
    )
    

# =============================================================================================================
# ======================================= CHECKPOINT 1 ========================================================
# =============================================================================================================
# =============================================================================================================
# ======================================= FROZEN MODEL SPECIFICATIONS =========================================
# =============================================================================================================

TCN_SCORE_THRESHOLD = 0.000693
GRU_SCORE_THRESHOLD = 0.000631
TSM_SCORE_THRESHOLD = 0.000775
ITR_SCORE_THRESHOLD = 0.000
DL_SCORE_THRESHOLD  = 0.000694

FORWARD_HOURS = 24

# =============================================================================================================
# ======================================= COMMON PARAMETERS / LOADING =========================================
# =============================================================================================================

EPOCHS = 40

TCN_SEED = 42
GRU_SEED = 11
TSM_SEED = 42
ITR_SEED = 42
DL_SEED = 42


with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/hyperparameters/tcn_best_params.pkl", "rb") as f:
    best_tcn_params = pickle.load(f)

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/hyperparameters/gru_best_params.pkl", "rb") as f:
    best_gru_params = pickle.load(f)

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/hyperparameters/tsm_best_params.pkl", "rb") as f:
    best_tsm_params = pickle.load(f)

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/hyperparameters/itr_best_params.pkl", "rb") as f:
    best_itr_params = pickle.load(f)

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/hyperparameters/dl_best_params.pkl", "rb") as f:
    best_dl_params = pickle.load(f)


sys.modules.pop("srcB.tcn_model_mfiv", None)
sys.modules.pop("srcB.gru_model_mfiv", None)
sys.modules.pop("srcB.tsm_model_mfiv", None)
sys.modules.pop("srcB.itr_model_mfiv", None)
sys.modules.pop("srcB.dl_model_mfiv", None)

from srcB.tcn_model_mfiv import tcn_mfiv, predict_tcn
from srcB.gru_model_mfiv import gru_mfiv, predict_gru
from srcB.tsm_model_mfiv import tsm_mfiv, predict_tsm
from srcB.itr_model_mfiv import itr_mfiv, predict_itr
from srcB.dl_model_mfiv import dl_mfiv, predict_dl


from srcB.market import Market

sys.modules.pop("srcB.trader_testing_tcn_v1", None)
from srcB.trader_testing_tcn_v1 import testing_execute_trading_strategy


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
    "itr_mfiv": {
        "action_col": "itr_action",
        "score_col": "itr_best_score",
        "threshold": ITR_SCORE_THRESHOLD
    },
    "dl_mfiv": {
        "action_col": "dl_action",
        "score_col": "dl_best_score",
        "threshold": DL_SCORE_THRESHOLD
    }
}

strategy_results = {}


# =============================================================================================================
# ======================================= FITTING 2022 ========================================================
# =============================================================================================================

FITTING_YEAR = 2022

X_fit_2022 = pd.concat([
    Xq_by_year[FITTING_YEAR][q]
    for q in QUARTERS
]).sort_index().copy()

garch_fit_2022 = pd.concat([
    outputs_by_year_OOS[FITTING_YEAR][q]["garch"]
    for q in QUARTERS
]).sort_index()

X_fit_2022["garch"] = garch_fit_2022.reindex(X_fit_2022.index)


_, tcn_bundle_2022, _ = tcn_mfiv(
    X_df=X_fit_2022,
    option_df=df_raw_by_year[FITTING_YEAR],
    time_df=d_btc_raw_by_year[FITTING_YEAR],
    sequence_length=168,
    forward_hours=FORWARD_HOURS,
    delta_target=0.20,
    channels=best_tcn_params["channels"],
    kernel_size=best_tcn_params["kernel_size"],
    dropout=best_tcn_params["dropout"],
    epochs=EPOCHS,
    batch_size=best_tcn_params["batch_size"],
    learning_rate=best_tcn_params["learning_rate"],
    weight_decay=best_tcn_params["weight_decay"],
    seed=TCN_SEED
)

_, gru_bundle_2022, _ = gru_mfiv(
    X_df=X_fit_2022,
    option_df=df_raw_by_year[FITTING_YEAR],
    time_df=d_btc_raw_by_year[FITTING_YEAR],
    sequence_length=168,
    forward_hours=FORWARD_HOURS,
    delta_target=0.20,
    hidden_size=best_gru_params["hidden_size"],
    num_layers=best_gru_params["num_layers"],
    dropout=best_gru_params["dropout"],
    epochs=EPOCHS,
    batch_size=best_gru_params["batch_size"],
    learning_rate=best_gru_params["learning_rate"],
    weight_decay=best_gru_params["weight_decay"],
    seed=GRU_SEED
)

_, tsm_bundle_2022, _ = tsm_mfiv(
    X_df=X_fit_2022,
    option_df=df_raw_by_year[FITTING_YEAR],
    time_df=d_btc_raw_by_year[FITTING_YEAR],
    sequence_length=168,
    forward_hours=FORWARD_HOURS,
    delta_target=0.20,
    num_blocks=best_tsm_params["num_blocks"],
    feature_hidden=best_tsm_params["feature_hidden"],
    dropout=best_tsm_params["dropout"],
    epochs=EPOCHS,
    batch_size=best_tsm_params["batch_size"],
    learning_rate=best_tsm_params["learning_rate"],
    weight_decay=best_tsm_params["weight_decay"],
    seed=TSM_SEED
)

_, itr_bundle_2022, _ = itr_mfiv(
    X_df=X_fit_2022,
    option_df=df_raw_by_year[FITTING_YEAR],
    time_df=d_btc_raw_by_year[FITTING_YEAR],
    sequence_length=168,
    forward_hours=FORWARD_HOURS,
    delta_target=0.20,
    d_model=best_itr_params["d_model"],
    n_heads=best_itr_params["n_heads"],
    e_layers=best_itr_params["e_layers"],
    d_ff=best_itr_params["d_ff"],
    dropout=best_itr_params["dropout"],
    epochs=EPOCHS,
    batch_size=best_itr_params["batch_size"],
    learning_rate=best_itr_params["learning_rate"],
    weight_decay=best_itr_params["weight_decay"],
    seed=ITR_SEED
)

_, dl_bundle_2022, _ = dl_mfiv(
    X_df=X_fit_2022,
    option_df=df_raw_by_year[FITTING_YEAR],
    time_df=d_btc_raw_by_year[FITTING_YEAR],
    sequence_length=168,
    forward_hours=FORWARD_HOURS,
    delta_target=0.20,
    moving_avg=best_dl_params["moving_avg"],
    individual=best_dl_params["individual"],
    epochs=EPOCHS,
    batch_size=best_dl_params["batch_size"],
    learning_rate=best_dl_params["learning_rate"],
    weight_decay=best_dl_params["weight_decay"],
    seed=DL_SEED
)


# =============================================================================================================
# ======================================= PREDICTION 2023 ======================================================
# =============================================================================================================

PREDICT_YEAR = 2023

X_prediction_2023 = pd.concat([
    Xq_by_year[y][q]
    for y in [2022, 2023]
    for q in QUARTERS
]).sort_index().copy()

garch_prediction_2023 = pd.concat([
    outputs_by_year_OOS[y][q]["garch"]
    for y in [2022, 2023]
    for q in QUARTERS
]).sort_index()

X_prediction_2023["garch"] = garch_prediction_2023.reindex(X_prediction_2023.index)


tcn_output_2023 = predict_tcn(tcn_bundle_2022, X_prediction_2023)
gru_output_2023 = predict_gru(gru_bundle_2022, X_prediction_2023)
tsm_output_2023 = predict_tsm(tsm_bundle_2022, X_prediction_2023)
itr_output_2023 = predict_itr(itr_bundle_2022, X_prediction_2023)
dl_output_2023 = predict_dl(dl_bundle_2022, X_prediction_2023)


tcn_output_2023 = tcn_output_2023[tcn_output_2023.index.year == PREDICT_YEAR].copy()
gru_output_2023 = gru_output_2023[gru_output_2023.index.year == PREDICT_YEAR].copy()
tsm_output_2023 = tsm_output_2023[tsm_output_2023.index.year == PREDICT_YEAR].copy()
itr_output_2023 = itr_output_2023[itr_output_2023.index.year == PREDICT_YEAR].copy()
dl_output_2023 = dl_output_2023[dl_output_2023.index.year == PREDICT_YEAR].copy()


prediction_outputs_2023 = [
    tcn_output_2023,
    gru_output_2023,
    tsm_output_2023,
    itr_output_2023,
    dl_output_2023
]

for q in QUARTERS:

    q_index = Xq_by_year[PREDICT_YEAR][q].index

    for model_output in prediction_outputs_2023:

        model_q = model_output.reindex(q_index)

        outputs_by_year_OOS[PREDICT_YEAR][q] = outputs_by_year_OOS[PREDICT_YEAR][q].drop(
            columns=model_q.columns,
            errors="ignore"
        )

        outputs_by_year_OOS[PREDICT_YEAR][q] = pd.concat(
            [outputs_by_year_OOS[PREDICT_YEAR][q], model_q],
            axis=1
        )

    outputs_by_year_OOS[PREDICT_YEAR][q]["rv"] = Xq_by_year[PREDICT_YEAR][q]["rv"].reindex(
        outputs_by_year_OOS[PREDICT_YEAR][q].index
    )

    outputs_by_year_OOS[PREDICT_YEAR][q]["MFIV_24"] = Xq_by_year[PREDICT_YEAR][q]["MFIV_24"].reindex(
        outputs_by_year_OOS[PREDICT_YEAR][q].index
    )


# =============================================================================================================
# ======================================= FITTING 2023 ========================================================
# =============================================================================================================

FITTING_YEAR = 2023

X_fit_2023 = pd.concat([
    Xq_by_year[FITTING_YEAR][q]
    for q in QUARTERS
]).sort_index().copy()

garch_fit_2023 = pd.concat([
    outputs_by_year_OOS[FITTING_YEAR][q]["garch"]
    for q in QUARTERS
]).sort_index()

X_fit_2023["garch"] = garch_fit_2023.reindex(X_fit_2023.index)


_, tcn_bundle_2023, _ = tcn_mfiv(
    X_df=X_fit_2023,
    option_df=df_raw_by_year[FITTING_YEAR],
    time_df=d_btc_raw_by_year[FITTING_YEAR],
    sequence_length=168,
    forward_hours=FORWARD_HOURS,
    delta_target=0.20,
    channels=best_tcn_params["channels"],
    kernel_size=best_tcn_params["kernel_size"],
    dropout=best_tcn_params["dropout"],
    epochs=EPOCHS,
    batch_size=best_tcn_params["batch_size"],
    learning_rate=best_tcn_params["learning_rate"],
    weight_decay=best_tcn_params["weight_decay"],
    seed=TCN_SEED
)

_, gru_bundle_2023, _ = gru_mfiv(
    X_df=X_fit_2023,
    option_df=df_raw_by_year[FITTING_YEAR],
    time_df=d_btc_raw_by_year[FITTING_YEAR],
    sequence_length=168,
    forward_hours=FORWARD_HOURS,
    delta_target=0.20,
    hidden_size=best_gru_params["hidden_size"],
    num_layers=best_gru_params["num_layers"],
    dropout=best_gru_params["dropout"],
    epochs=EPOCHS,
    batch_size=best_gru_params["batch_size"],
    learning_rate=best_gru_params["learning_rate"],
    weight_decay=best_gru_params["weight_decay"],
    seed=GRU_SEED
)

_, tsm_bundle_2023, _ = tsm_mfiv(
    X_df=X_fit_2023,
    option_df=df_raw_by_year[FITTING_YEAR],
    time_df=d_btc_raw_by_year[FITTING_YEAR],
    sequence_length=168,
    forward_hours=FORWARD_HOURS,
    delta_target=0.20,
    num_blocks=best_tsm_params["num_blocks"],
    feature_hidden=best_tsm_params["feature_hidden"],
    dropout=best_tsm_params["dropout"],
    epochs=EPOCHS,
    batch_size=best_tsm_params["batch_size"],
    learning_rate=best_tsm_params["learning_rate"],
    weight_decay=best_tsm_params["weight_decay"],
    seed=TSM_SEED
)

_, itr_bundle_2023, _ = itr_mfiv(
    X_df=X_fit_2023,
    option_df=df_raw_by_year[FITTING_YEAR],
    time_df=d_btc_raw_by_year[FITTING_YEAR],
    sequence_length=168,
    forward_hours=FORWARD_HOURS,
    delta_target=0.20,
    d_model=best_itr_params["d_model"],
    n_heads=best_itr_params["n_heads"],
    e_layers=best_itr_params["e_layers"],
    d_ff=best_itr_params["d_ff"],
    dropout=best_itr_params["dropout"],
    epochs=EPOCHS,
    batch_size=best_itr_params["batch_size"],
    learning_rate=best_itr_params["learning_rate"],
    weight_decay=best_itr_params["weight_decay"],
    seed=ITR_SEED
)

_, dl_bundle_2023, _ = dl_mfiv(
    X_df=X_fit_2023,
    option_df=df_raw_by_year[FITTING_YEAR],
    time_df=d_btc_raw_by_year[FITTING_YEAR],
    sequence_length=168,
    forward_hours=FORWARD_HOURS,
    delta_target=0.20,
    moving_avg=best_dl_params["moving_avg"],
    individual=best_dl_params["individual"],
    epochs=EPOCHS,
    batch_size=best_dl_params["batch_size"],
    learning_rate=best_dl_params["learning_rate"],
    weight_decay=best_dl_params["weight_decay"],
    seed=DL_SEED
)


# =============================================================================================================
# ======================================= PREDICTION 2024 ======================================================
# =============================================================================================================

PREDICT_YEAR = 2024

X_prediction_2024 = pd.concat([
    Xq_by_year[y][q]
    for y in [2023, 2024]
    for q in QUARTERS
]).sort_index().copy()

garch_prediction_2024 = pd.concat([
    outputs_by_year_OOS[y][q]["garch"]
    for y in [2023, 2024]
    for q in QUARTERS
]).sort_index()

X_prediction_2024["garch"] = garch_prediction_2024.reindex(X_prediction_2024.index)


tcn_output_2024 = predict_tcn(tcn_bundle_2023, X_prediction_2024)
gru_output_2024 = predict_gru(gru_bundle_2023, X_prediction_2024)
tsm_output_2024 = predict_tsm(tsm_bundle_2023, X_prediction_2024)
itr_output_2024 = predict_itr(itr_bundle_2023, X_prediction_2024)
dl_output_2024 = predict_dl(dl_bundle_2023, X_prediction_2024)


tcn_output_2024 = tcn_output_2024[tcn_output_2024.index.year == PREDICT_YEAR].copy()
gru_output_2024 = gru_output_2024[gru_output_2024.index.year == PREDICT_YEAR].copy()
tsm_output_2024 = tsm_output_2024[tsm_output_2024.index.year == PREDICT_YEAR].copy()
itr_output_2024 = itr_output_2024[itr_output_2024.index.year == PREDICT_YEAR].copy()
dl_output_2024 = dl_output_2024[dl_output_2024.index.year == PREDICT_YEAR].copy()


prediction_outputs_2024 = [
    tcn_output_2024,
    gru_output_2024,
    tsm_output_2024,
    itr_output_2024,
    dl_output_2024
]

for q in QUARTERS:

    q_index = Xq_by_year[PREDICT_YEAR][q].index

    for model_output in prediction_outputs_2024:

        model_q = model_output.reindex(q_index)

        outputs_by_year_OOS[PREDICT_YEAR][q] = outputs_by_year_OOS[PREDICT_YEAR][q].drop(
            columns=model_q.columns,
            errors="ignore"
        )

        outputs_by_year_OOS[PREDICT_YEAR][q] = pd.concat(
            [outputs_by_year_OOS[PREDICT_YEAR][q], model_q],
            axis=1
        )

    outputs_by_year_OOS[PREDICT_YEAR][q]["rv"] = Xq_by_year[PREDICT_YEAR][q]["rv"].reindex(
        outputs_by_year_OOS[PREDICT_YEAR][q].index
    )

    outputs_by_year_OOS[PREDICT_YEAR][q]["MFIV_24"] = Xq_by_year[PREDICT_YEAR][q]["MFIV_24"].reindex(
        outputs_by_year_OOS[PREDICT_YEAR][q].index
    )


# =============================================================================================================
# ======================================= FITTING 2024 ========================================================
# =============================================================================================================

FITTING_YEAR = 2024

X_fit_2024 = pd.concat([
    Xq_by_year[FITTING_YEAR][q]
    for q in QUARTERS
]).sort_index().copy()

garch_fit_2024 = pd.concat([
    outputs_by_year_OOS[FITTING_YEAR][q]["garch"]
    for q in QUARTERS
]).sort_index()

X_fit_2024["garch"] = garch_fit_2024.reindex(X_fit_2024.index)


_, tcn_bundle_2024, _ = tcn_mfiv(
    X_df=X_fit_2024,
    option_df=df_raw_by_year[FITTING_YEAR],
    time_df=d_btc_raw_by_year[FITTING_YEAR],
    sequence_length=168,
    forward_hours=FORWARD_HOURS,
    delta_target=0.20,
    channels=best_tcn_params["channels"],
    kernel_size=best_tcn_params["kernel_size"],
    dropout=best_tcn_params["dropout"],
    epochs=EPOCHS,
    batch_size=best_tcn_params["batch_size"],
    learning_rate=best_tcn_params["learning_rate"],
    weight_decay=best_tcn_params["weight_decay"],
    seed=TCN_SEED
)

_, gru_bundle_2024, _ = gru_mfiv(
    X_df=X_fit_2024,
    option_df=df_raw_by_year[FITTING_YEAR],
    time_df=d_btc_raw_by_year[FITTING_YEAR],
    sequence_length=168,
    forward_hours=FORWARD_HOURS,
    delta_target=0.20,
    hidden_size=best_gru_params["hidden_size"],
    num_layers=best_gru_params["num_layers"],
    dropout=best_gru_params["dropout"],
    epochs=EPOCHS,
    batch_size=best_gru_params["batch_size"],
    learning_rate=best_gru_params["learning_rate"],
    weight_decay=best_gru_params["weight_decay"],
    seed=GRU_SEED
)

_, tsm_bundle_2024, _ = tsm_mfiv(
    X_df=X_fit_2024,
    option_df=df_raw_by_year[FITTING_YEAR],
    time_df=d_btc_raw_by_year[FITTING_YEAR],
    sequence_length=168,
    forward_hours=FORWARD_HOURS,
    delta_target=0.20,
    num_blocks=best_tsm_params["num_blocks"],
    feature_hidden=best_tsm_params["feature_hidden"],
    dropout=best_tsm_params["dropout"],
    epochs=EPOCHS,
    batch_size=best_tsm_params["batch_size"],
    learning_rate=best_tsm_params["learning_rate"],
    weight_decay=best_tsm_params["weight_decay"],
    seed=TSM_SEED
)

_, itr_bundle_2024, _ = itr_mfiv(
    X_df=X_fit_2024,
    option_df=df_raw_by_year[FITTING_YEAR],
    time_df=d_btc_raw_by_year[FITTING_YEAR],
    sequence_length=168,
    forward_hours=FORWARD_HOURS,
    delta_target=0.20,
    d_model=best_itr_params["d_model"],
    n_heads=best_itr_params["n_heads"],
    e_layers=best_itr_params["e_layers"],
    d_ff=best_itr_params["d_ff"],
    dropout=best_itr_params["dropout"],
    epochs=EPOCHS,
    batch_size=best_itr_params["batch_size"],
    learning_rate=best_itr_params["learning_rate"],
    weight_decay=best_itr_params["weight_decay"],
    seed=ITR_SEED
)

_, dl_bundle_2024, _ = dl_mfiv(
    X_df=X_fit_2024,
    option_df=df_raw_by_year[FITTING_YEAR],
    time_df=d_btc_raw_by_year[FITTING_YEAR],
    sequence_length=168,
    forward_hours=FORWARD_HOURS,
    delta_target=0.20,
    moving_avg=best_dl_params["moving_avg"],
    individual=best_dl_params["individual"],
    epochs=EPOCHS,
    batch_size=best_dl_params["batch_size"],
    learning_rate=best_dl_params["learning_rate"],
    weight_decay=best_dl_params["weight_decay"],
    seed=DL_SEED
)


# =============================================================================================================
# ======================================= PREDICTION 2025 ======================================================
# =============================================================================================================

PREDICT_YEAR = 2025

X_prediction_2025 = pd.concat([
    Xq_by_year[y][q]
    for y in [2024, 2025]
    for q in QUARTERS
]).sort_index().copy()

garch_prediction_2025 = pd.concat([
    outputs_by_year_OOS[y][q]["garch"]
    for y in [2024, 2025]
    for q in QUARTERS
]).sort_index()

X_prediction_2025["garch"] = garch_prediction_2025.reindex(X_prediction_2025.index)


tcn_output_2025 = predict_tcn(tcn_bundle_2024, X_prediction_2025)
gru_output_2025 = predict_gru(gru_bundle_2024, X_prediction_2025)
tsm_output_2025 = predict_tsm(tsm_bundle_2024, X_prediction_2025)
itr_output_2025 = predict_itr(itr_bundle_2024, X_prediction_2025)
dl_output_2025 = predict_dl(dl_bundle_2024, X_prediction_2025)


tcn_output_2025 = tcn_output_2025[tcn_output_2025.index.year == PREDICT_YEAR].copy()
gru_output_2025 = gru_output_2025[gru_output_2025.index.year == PREDICT_YEAR].copy()
tsm_output_2025 = tsm_output_2025[tsm_output_2025.index.year == PREDICT_YEAR].copy()
itr_output_2025 = itr_output_2025[itr_output_2025.index.year == PREDICT_YEAR].copy()
dl_output_2025 = dl_output_2025[dl_output_2025.index.year == PREDICT_YEAR].copy()


prediction_outputs_2025 = [
    tcn_output_2025,
    gru_output_2025,
    tsm_output_2025,
    itr_output_2025,
    dl_output_2025
]

for q in QUARTERS:

    q_index = Xq_by_year[PREDICT_YEAR][q].index

    for model_output in prediction_outputs_2025:

        model_q = model_output.reindex(q_index)

        outputs_by_year_OOS[PREDICT_YEAR][q] = outputs_by_year_OOS[PREDICT_YEAR][q].drop(
            columns=model_q.columns,
            errors="ignore"
        )

        outputs_by_year_OOS[PREDICT_YEAR][q] = pd.concat(
            [outputs_by_year_OOS[PREDICT_YEAR][q], model_q],
            axis=1
        )

    outputs_by_year_OOS[PREDICT_YEAR][q]["rv"] = Xq_by_year[PREDICT_YEAR][q]["rv"].reindex(
        outputs_by_year_OOS[PREDICT_YEAR][q].index
    )

    outputs_by_year_OOS[PREDICT_YEAR][q]["MFIV_24"] = Xq_by_year[PREDICT_YEAR][q]["MFIV_24"].reindex(
        outputs_by_year_OOS[PREDICT_YEAR][q].index
    )


# =============================================================================================================
# ======================================= GLOBAL OOS TRADER ===================================================
# =============================================================================================================

for PREDICT_YEAR in PREDICT_YEARS:

    for MODEL, config in MODEL_CONFIGS.items():

        outputs_trader = {
            PREDICT_YEAR: {
                q: outputs_by_year_OOS[PREDICT_YEAR][q].copy()
                for q in QUARTERS
            }
        }

        for q in QUARTERS:

            outputs_trader[PREDICT_YEAR][q]["tcn_action"] = outputs_trader[PREDICT_YEAR][q][
                config["action_col"]
            ]

            outputs_trader[PREDICT_YEAR][q]["tcn_best_score"] = outputs_trader[PREDICT_YEAR][q][
                config["score_col"]
            ]


        trader_cache = {
            PREDICT_YEAR: pd.concat([
                outputs_trader[PREDICT_YEAR][q]
                for q in QUARTERS
            ]).sort_index()
        }


        market = Market(
            option_df=df_raw_by_year[PREDICT_YEAR],
            time_df=d_btc_raw_by_year[PREDICT_YEAR],
            starting_cash=100_000,
            im_multiplier=0.15,
            mm_buffer_rate=0.05,
        )


        snapshots, greek_tests = testing_execute_trading_strategy(
            bt_year=PREDICT_YEAR,
            bt_quarter=None,
            model="tcn_mfiv",
            outputs_by_year=outputs_trader,
            market=market,
            trader_cache=trader_cache,
            run_by="year",
            tcn_score_threshold=config["threshold"],
            forward_hours=FORWARD_HOURS
        )

        history_key = f"{MODEL}_{str(PREDICT_YEAR)[2:]}"

        strategy_results[history_key] = {
            "equity_curve": market.get_equity_history(),
            "snapshots": snapshots,
            "greek_tests": greek_tests,
            "orders": list(market.orders),
            "transactions": list(market.transactions),
            "final_cash": market.cash,
            "required_margin": market.get_total_required_margin(),
            "free_liquidity": market.get_free_liquidity(),
        }


print(strategy_results.keys())


# =============================================================================================================
# ======================================= SAVE OOS OUTPUTS =====================================================
# =============================================================================================================
os.makedirs("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/OOS/OOS_GLOBAL", exist_ok = True)

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/OOS/OOS_GLOBAL/outputs_by_year.pkl", "wb") as f:
    pickle.dump(outputs_by_year_OOS, f)

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/OOS/OOS_GLOBAL/strategy_results.pkl", "wb") as f:
    pickle.dump(strategy_results, f)


speak_done()