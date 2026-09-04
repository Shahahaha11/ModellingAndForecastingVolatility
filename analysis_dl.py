# ANALYSIS DLINEAR
# SEARCH WORDS :
# YEARLY_DL_ESTIMATION
# DL_FITTED_TARGET_SPACE_PERFORMANCE
# DL_SCORE_CALIBRATION
# DL_THRESHOLD_CALIBRATION
# DL_THRESHOLD_TRADER_CALIBRATION

import pandas as pd
import numpy as np
import warnings
import os, sys
import pickle

from pathlib import Path
from arch.utility.exceptions import ConvergenceWarning


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


sys.path.insert(0,"/Users/shah/CODE_BOOK_4/THESIS_WORKING/THESIS_2")
sys.path.insert(0,"/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/MFIV_calibration")

os.chdir("/Users/shah/CODE_BOOK_4/THESIS_WORKING/THESIS_2")

sys.modules.pop("srcB", None)


# =============================================================================================================
# ======================================= LOAD BASE OUTPUTS ===================================================
# =============================================================================================================

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/outputs_by_year/outputs_by_year_04.pkl","rb") as f:
    outputs_by_year = pickle.load(f)


# =============================================================================================================
# ======================================= IMPORTS FROM SRC ====================================================
# =============================================================================================================

from srcB.market import *
from srcB.helpers import *


# =============================================================================================================
# ======================================= READ D_BTC AND DF ===================================================
# =============================================================================================================

DATA_DIR = Path("../data/rawData")

YEARS = [2019, 2020, 2021]
QUARTERS = ["q1", "q2", "q3", "q4"]

d_btc_raw_by_year = {}
df_raw_by_year = {}

for y in YEARS:
    d_btc_raw_by_year[y] = pd.read_parquet(DATA_DIR/f"btc_underlying_minutely_{y}.parquet")
    df_raw_by_year[y] = pd.read_parquet(DATA_DIR/f"derivative_data_{y}.parquet")


# =============================================================================================================
# ======================================= SET INDEX TO DATETIME ===============================================
# =============================================================================================================

for y in YEARS:
    d_btc_raw_by_year[y] = d_btc_raw_by_year[y].set_index(pd.to_datetime(d_btc_raw_by_year[y]["date_time"]))
    df_raw_by_year[y] = df_raw_by_year[y].set_index(pd.to_datetime(df_raw_by_year[y]["date_time"]))


# =============================================================================================================
# ======================================= LOAD X_BY_YEAR ======================================================
# =============================================================================================================

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/x_by_year_fut/x_by_year.pkl", "rb") as f:
    X_by_year = pd.read_pickle(f)

for y in YEARS:
    X_by_year[y] = X_by_year[y].copy()
    X_by_year[y]["rv"] = X_by_year[y]["rt"] ** 2

Xq_by_year = {}

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
        Xq_by_year[y][q] = Xq_by_year[y][q].loc[:, ~Xq_by_year[y][q].columns.duplicated(keep="first")]


# =============================================================================================================
# ======================================= BORROW RFR COLUMN ===================================================
# =============================================================================================================

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/ready_data/d_btc_by_year_all.pkl", "rb") as f:
    d_btc_by_year = pickle.load(f)

for y in YEARS:
    d_btc_raw_by_year[y]["rfr"] = d_btc_by_year[y]["rfr"].reindex(d_btc_raw_by_year[y].index, method="ffill")


# =============================================================================================================
# ======================================= CHECKPOINT 1 ========================================================
# =============================================================================================================


# =============================================================================================================
# ======================================= DLINEAR PARAMETERS ==================================================
# =============================================================================================================

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/hyperparameters/dl_best_params.pkl", "rb") as f:
    best_dl_params = pickle.load(f)

VAL_FRACTION = 0.20
EPOCHS = 40
PATIENCE = 6
SEED = 42

# =============================================================================================================
# ======================================= YEARLY_DL_ESTIMATION ================================================
# =============================================================================================================

sys.modules.pop("srcB.dl_model", None)
from srcB.dl_model import *

dl_bundle_by_year = {}
dl_targets_by_year = {}
dl_output_by_year = {}

for y in YEARS:
    X_df = pd.concat([Xq_by_year[y][q] for q in QUARTERS]).sort_index().copy()
    garch_df = pd.concat([outputs_by_year[y][q]["garch"] for q in QUARTERS]).sort_index()
    X_df["garch"] = garch_df.reindex(X_df.index)

    dl_output, dl_bundle, dl_targets = dl_mfiv(
        X_df=X_df,
        option_df=df_raw_by_year[y],
        time_df=d_btc_raw_by_year[y],
        sequence_length=168,
        forward_hours=24,
        delta_target=0.20,
        val_fraction=VAL_FRACTION,
        moving_avg=best_dl_params["moving_avg"],
        individual=best_dl_params["individual"],
        epochs=EPOCHS,
        batch_size=best_dl_params["batch_size"],
        learning_rate=best_dl_params["learning_rate"],
        weight_decay=best_dl_params["weight_decay"],
        patience=PATIENCE,
        seed=SEED
    )

    dl_output_by_year[y] = dl_output
    dl_bundle_by_year[y] = dl_bundle
    dl_targets_by_year[y] = dl_targets

    for q in QUARTERS:
        q_index = Xq_by_year[y][q].index
        dl_q = dl_output.reindex(q_index)
        outputs_by_year[y][q] = outputs_by_year[y][q].drop(columns=dl_q.columns, errors="ignore")
        outputs_by_year[y][q] = pd.concat([outputs_by_year[y][q], dl_q], axis=1)


# =============================================================================================================
# ================================ DL_FITTED_TARGET_SPACE_PERFORMANCE =========================================
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
        dl_output_by_year[y][["dl_action"]],
        dl_targets_by_year[y][target_cols]
    ], axis=1, sort=False).dropna()

    realized = pd.Series(0.0, index=test.index)

    for action in ["long_straddle", "short_straddle", "long_strangle", "short_strangle"]:
        mask = test["dl_action"] == action
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
# ======================================= DL_SCORE_CALIBRATION ================================================
# =============================================================================================================

rows = []

for y in YEARS:
    test = pd.concat([
        dl_output_by_year[y][["dl_action", "dl_best_score"]],
        dl_targets_by_year[y][target_cols]
    ], axis=1, sort=False).dropna()

    test = test[test["dl_action"] != "no_trade"].copy()
    test["target_pnl"] = 0.0

    for action in ["long_straddle", "short_straddle", "long_strangle", "short_strangle"]:
        m = test["dl_action"] == action
        test.loc[m, "target_pnl"] = test.loc[m, f"target_{action}"]

    test["score_bin"] = pd.qcut(
        test["dl_best_score"],
        5,
        labels=False,
        duplicates="drop"
    )

    x = (
        test.groupby("score_bin")
        .agg(
            n=("target_pnl", "size"),
            mean_score=("dl_best_score", "mean"),
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


# =============================================================================================================
# ======================================= DL_THRESHOLD_CALIBRATION ============================================
# =============================================================================================================

tests = {}

for y in YEARS:
    x = pd.concat([
        dl_output_by_year[y][["dl_action", "dl_best_score"]],
        dl_targets_by_year[y][target_cols]
    ], axis=1, sort=False).dropna()

    x = x[x["dl_action"] != "no_trade"].copy()
    x["target_pnl"] = 0.0

    for action in ["long_straddle", "short_straddle", "long_strangle", "short_strangle"]:
        m = x["dl_action"] == action
        x.loc[m, "target_pnl"] = x.loc[m, f"target_{action}"]

    tests[y] = x

scores_all = pd.concat([tests[y]["dl_best_score"] for y in YEARS])

rows = []

for q in [0, .20, .40, .50, .60, .70, .80, .85, .90]:

    threshold = scores_all.quantile(q)

    for y in YEARS:
        x = tests[y][tests[y]["dl_best_score"] >= threshold]
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


# =============================================================================================================
# ======================================= DL SCORE DISTRIBUTION ===============================================
# =============================================================================================================

for y in YEARS:
    x = tests[y]["dl_best_score"]

    print(
        y,
        "n:", len(x),
        "min:", x.min(),
        "q25:", x.quantile(.25),
        "q50:", x.quantile(.50),
        "q75:", x.quantile(.75),
        "q90:", x.quantile(.90),
        "max:", x.max()
    )


# =============================================================================================================
# ======================================= SAVE DL BUNDLES =====================================================
# =============================================================================================================

os.makedirs("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/dl", exist_ok=True)
with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/dl/dl_bundle_by_year.pkl", "wb") as f:
    pickle.dump(dl_bundle_by_year, f)


# =============================================================================================================
# ======================================= PREPARE TRADER INPUT ================================================
# =============================================================================================================

for y in YEARS:
    for q in QUARTERS:
        outputs_by_year[y][q]["rv"] = Xq_by_year[y][q]["rv"].reindex(
            outputs_by_year[y][q].index
        )
        outputs_by_year[y][q]["MFIV_24"] = Xq_by_year[y][q]["MFIV_24"].reindex(
            outputs_by_year[y][q].index
        )


# =============================================================================================================
# ======================================= DL TRADER ADAPTER ===================================================
# =============================================================================================================
# DLINEAR signals are not altered;
# it only copies them into column names the existing trader expects.
for y in YEARS:
    for q in QUARTERS:
        outputs_by_year[y][q]["tcn_action"] = outputs_by_year[y][q]["dl_action"]
        outputs_by_year[y][q]["tcn_best_score"] = outputs_by_year[y][q]["dl_best_score"]


# =============================================================================================================
# ======================================= CACHE FOR WARMUP ====================================================
# =============================================================================================================

CACHE_YEARS = [2019, 2020, 2021]

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


# =============================================================================================================
# ================================ DL_THRESHOLD_TRADER_CALIBRATION ============================================
# =============================================================================================================

sys.modules.pop("srcB.trader_testing_tcn", None)
from srcB.trader_testing_tcn import testing_execute_trading_strategy

THRESHOLDS = [0.0]

for q in [.25, .50, .75, .90]:
    threshold = min(
        tests[y]["dl_best_score"].quantile(q)
        for y in YEARS
    )
    THRESHOLDS.append(float(threshold))

THRESHOLDS = sorted(set(THRESHOLDS))

print("DLinear trader thresholds:", THRESHOLDS)

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

# DLinear :0.005388