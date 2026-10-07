# ================= CURRENT FOLD LOGIC =================

# 40% train -> exact target-end purge -> next 20%
# 60% train -> exact target-end purge -> next 20%
# 80% train -> exact target-end purge -> final 20%
#
# A training observation is retained only if every realized option target
# used for that observation is fully resolved before validation begins.

# =============================================================================================================
# ======================================= HYPERPARAMETER TUNING ===============================================
# =============================================================================================================

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import optuna

os.chdir("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/MFIV_calibration")


# =============================================================================================================
# ======================================= COMMON PARAMETERS ===================================================
# =============================================================================================================

YEARS = [2019, 2020, 2021]
QUARTERS = ["q1", "q2", "q3", "q4"]

SEQUENCE_LENGTH = 168
FORWARD_HOURS = 24
DELTA_TARGET = 0.20

VAL_FRACTION = 0.20
TRAIN_FRACTIONS = [0.40, 0.60, 0.80]

EPOCHS = 40
PATIENCE = 6
SEED = 42

N_TRIALS = 80


# =============================================================================================================
# ======================================= DATA PATHS ==========================================================
# =============================================================================================================

DATA_DIR = Path("../data/rawData")

OUTPUTS_PATH = ("../data/IS/outputs_by_year/outputs_by_year_01.pkl")

X_BY_YEAR_PATH = ("../data/IS/x_by_year_fut/x_by_year.pkl")

RFR_PATH = ("../data/IS/ready_data/d_btc_by_year_all.pkl")

HYPERPARAM_DIR = ("../data/IS/hyperparameters")


# =============================================================================================================
# ======================================= COMMON FEATURE / TARGET IMPORTS =====================================
# =============================================================================================================

sys.modules.pop("srcB.tcn_model_mfiv", None)

from srcB.tcn_model_mfiv import (
    build_tcn_features as build_common_features,
    build_forward_targets as build_common_targets,
    make_training_sequences as make_common_sequences,
    SmallCausalTCN,
    _scalers as tcn_scalers,
    _train as train_tcn,
    _get_device as get_tcn_device,
)


# =============================================================================================================
# ======================================= LOAD IS DATA ========================================================
# =============================================================================================================

def load_is_data():

    with open(OUTPUTS_PATH, "rb") as f:
        outputs_by_year = pickle.load(f)

    X_by_year = pd.read_pickle(X_BY_YEAR_PATH)

    with open(RFR_PATH, "rb") as f:
        d_btc_by_year = pickle.load(f)

    d_btc_raw_by_year = {}
    df_raw_by_year = {}

    for y in YEARS:

        d_btc_raw_by_year[y] = pd.read_parquet(
            DATA_DIR / f"btc_underlying_minutely_{y}.parquet"
        )

        df_raw_by_year[y] = pd.read_parquet(
            DATA_DIR / f"derivative_data_{y}.parquet"
        )

        d_btc_raw_by_year[y] = d_btc_raw_by_year[y].set_index(
            pd.to_datetime(d_btc_raw_by_year[y]["date_time"])
        )

        df_raw_by_year[y] = df_raw_by_year[y].set_index(
            pd.to_datetime(df_raw_by_year[y]["date_time"])
        )

        d_btc_raw_by_year[y]["rfr"] = d_btc_by_year[y]["rfr"].reindex(
            d_btc_raw_by_year[y].index,
            method="ffill"
        )

    return (
        X_by_year,
        outputs_by_year,
        d_btc_raw_by_year,
        df_raw_by_year,
    )


# =============================================================================================================
# ======================================= PREPARE ONE MODEL YEAR ==============================================
# =============================================================================================================
def prepare_model_year(
    year,
    X_by_year,
    outputs_by_year,
    d_btc_raw_by_year,
    df_raw_by_year,
):

    X_df = X_by_year[year].copy().sort_index()

    X_df["rv"] = X_df["rt"] ** 2

    for H in [12, 24, 72, 168, 336]:
        X_df[f"MFIV_{H}"] *= H / 8760.0

    garch = pd.concat([
        outputs_by_year[year][q]["garch"]
        for q in QUARTERS
    ]).sort_index()

    X_df["garch"] = garch.reindex(X_df.index)

    features = build_common_features(X_df)

    targets = build_common_targets(
        option_df=df_raw_by_year[year],
        time_df=d_btc_raw_by_year[year],
        times=features.dropna().index,
        forward_hours=FORWARD_HOURS,
        delta_target=DELTA_TARGET,
    )

    X, y, idx = make_common_sequences(
        features,
        targets,
        sequence_length=SEQUENCE_LENGTH,
    )

    exit_cols = [
        "long_straddle_exit_time",
        "short_straddle_exit_time",
        "long_strangle_exit_time",
        "short_strangle_exit_time",
    ]

    target_end = (targets.loc[idx, exit_cols].max(axis=1).reindex(idx))

    return {
        "X": X,
        "y": y,
        "idx": idx,
        "target_end": target_end,
        "n_features": features.shape[1],
    }

# =============================================================================================================
# ======================================= PREPARE ALL MODEL YEARS =============================================
# =============================================================================================================

def prepare_model_data(
    X_by_year,
    outputs_by_year,
    d_btc_raw_by_year,
    df_raw_by_year,
    model_name,
):

    model_data = {}

    for y in YEARS:

        print(f"Preparing {model_name} data:", y)

        model_data[y] = prepare_model_year(
            year=y,
            X_by_year=X_by_year,
            outputs_by_year=outputs_by_year,
            d_btc_raw_by_year=d_btc_raw_by_year,
            df_raw_by_year=df_raw_by_year,
        )

        print("Sequences:", len(model_data[y]["idx"]))
        print("Features:", model_data[y]["n_features"])

    return model_data


# =============================================================================================================
# ======================================= ROLLING CHRONOLOGICAL FOLD ==========================================
# =============================================================================================================

def make_rolling_fold(data, train_fraction):

    X = data["X"]
    y = data["y"]
    idx = data["idx"]
    target_end = data["target_end"]

    val_start_i = int(len(idx) * train_fraction)
    val_end_i = int(len(idx) * (train_fraction + VAL_FRACTION))

    val_start = idx[val_start_i]

    train_mask = (target_end < val_start).to_numpy()

    if val_end_i < len(idx):

        val_end = idx[val_end_i]

        val_mask = (
            (idx >= val_start)
            & (idx < val_end)
        )

    else:
        val_mask = idx >= val_start

    return {
        "X_train": X[train_mask],
        "y_train": y[train_mask],
        "X_val": X[val_mask],
        "y_val": y[val_mask],
    }

# =============================================================================================================
# ======================================= FIT ONE MODEL FOLD ==================================================
# =============================================================================================================

def fit_model_fold(
    fold,
    model,
    params,
    scalers,
    train,
    device,
    seed,
):

    np.random.seed(seed)
    torch.manual_seed(seed)

    X_train = fold["X_train"]
    y_train = fold["y_train"]

    X_val = fold["X_val"]
    y_val = fold["y_val"]

    x_mean, x_std, y_mean, y_std = scalers(
        X_train,
        y_train
    )

    X_train = ((X_train - x_mean) / x_std).astype(np.float32)
    X_val = ((X_val - x_mean) / x_std).astype(np.float32)

    y_train = ((y_train - y_mean) / y_std).astype(np.float32)
    y_val = ((y_val - y_mean) / y_std).astype(np.float32)

    model, best_epoch, history = train(
        model=model,
        X=X_train,
        y=y_train,
        device=device,
        epochs=EPOCHS,
        batch_size=params["batch_size"],
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        X_val=X_val,
        y_val=y_val,
        patience=PATIENCE,
    )

    return history["val_loss"].min(), int(best_epoch)


# =============================================================================================================
# ======================================= LOAD DATA ONCE ======================================================
# =============================================================================================================

(
    X_by_year,
    outputs_by_year,
    d_btc_raw_by_year,
    df_raw_by_year,
) = load_is_data()



# =============================================================================================================
# ======================================= GRU IMPORTS =========================================================
# =============================================================================================================

sys.modules.pop("srcB.gru_model_mfiv", None)

from srcB.gru_model_mfiv import (
    SmallGRU,
    _scalers as gru_scalers,
    _train as train_gru,
    _get_device as get_gru_device,
)

GRU_SEED = 11


# =============================================================================================================
# ======================================= GRU SEARCH SPACE ====================================================
# =============================================================================================================

def suggest_gru_params(trial):

    hidden_size = trial.suggest_categorical(
        "hidden_size",
        [8, 16, 32, 64, 128] # EDITED
    )

    num_layers = trial.suggest_categorical(
        "num_layers",
        [1, 2, 3, 4, 6, 8, 10, 12] # EDITED
    )

    if num_layers == 1:
        dropout = 0.0
    else:
        dropout = trial.suggest_categorical(
            "dropout",
            [0.0, 0.10, 0.20, 0.30, 0.40] # EDITED
        )

    learning_rate = trial.suggest_float(
        "learning_rate",
        2e-4, # edited 0.00002
        1e-2, # edited 0.005
        log=True
    )

    weight_decay = trial.suggest_categorical(
        "weight_decay",
        [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [8, 16, 32, 64, 128, 256]
        
    )

    return {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
    }


# =============================================================================================================
# ======================================= EVALUATE ONE GRU TRIAL ==============================================
# =============================================================================================================

def evaluate_gru_trial(trial, gru_data):

    params = suggest_gru_params(trial)

    all_losses = []
    epochs_by_year = {y: [] for y in YEARS}

    resources = [1, 2, 4]

    for resource, train_fraction in zip(resources, TRAIN_FRACTIONS):

        round_losses = []

        for y in YEARS:

            fold = make_rolling_fold(
                gru_data[y],
                train_fraction
            )

            np.random.seed(GRU_SEED)
            torch.manual_seed(GRU_SEED)

            model = SmallGRU(
                n_features=gru_data[y]["n_features"],
                hidden_size=params["hidden_size"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
            )

            val_mse, best_epoch = fit_model_fold(
                fold=fold,
                model=model,
                params=params,
                scalers=gru_scalers,
                train=train_gru,
                device=get_gru_device(),
                seed=GRU_SEED,
            )

            round_losses.append(val_mse)
            epochs_by_year[y].append(best_epoch)

        all_losses.extend(round_losses)

        mean_val_mse = float(np.mean(all_losses))

        trial.report(
            mean_val_mse,
            step=resource
        )

        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr(
        "epochs_by_year",
        {
            str(y): int(np.median(epochs_by_year[y]))
            for y in YEARS
        },
    )

    return mean_val_mse


# =============================================================================================================
# ======================================= RUN GRU TUNING ======================================================
# =============================================================================================================

def tune_gru(gru_data):

    sampler = optuna.samplers.TPESampler(
        seed=GRU_SEED,
        n_startup_trials=10
    )

    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1,
        reduction_factor=2
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner
    )

    study.optimize(
        lambda trial: evaluate_gru_trial(
            trial,
            gru_data
        ),
        n_trials=N_TRIALS
    )

    return study


# =============================================================================================================
# ======================================= GRU MAIN ============================================================
# =============================================================================================================

gru_data = prepare_model_data(
    X_by_year=X_by_year,
    outputs_by_year=outputs_by_year,
    d_btc_raw_by_year=d_btc_raw_by_year,
    df_raw_by_year=df_raw_by_year,
    model_name="GRU",
)

gru_study = tune_gru(gru_data)

print("\nBEST GRU VALIDATION MSE")
print(gru_study.best_value)

print("\nBEST GRU PARAMETERS")
print(gru_study.best_params)

best_gru_params = gru_study.best_params.copy()

if best_gru_params["num_layers"] == 1:
    best_gru_params["dropout"] = 0.0

best_gru_params["epochs_by_year"] = {
int(y): int(epoch)
    for y, epoch in gru_study.best_trial.user_attrs["epochs_by_year"].items()
}

print("\nGRU EPOCHS BY YEAR")
print(best_gru_params["epochs_by_year"])


# =============================================================================================================
# ======================================= SAVE GRU HPO RESULTS ================================================
# =============================================================================================================

expand_index = "_e01"
with open(f"{HYPERPARAM_DIR}/gru_study{expand_index}.pkl", "wb") as f:
    pickle.dump(gru_study, f)

with open(f"{HYPERPARAM_DIR}/gru_best_params{expand_index}.pkl", "wb") as f:
    pickle.dump(best_gru_params, f)
