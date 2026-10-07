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

OUTPUTS_PATH = (
    "../data/IS/outputs_by_year/outputs_by_year_01.pkl"
)

X_BY_YEAR_PATH = (
    "../data/IS/x_by_year_fut/x_by_year.pkl"
)

RFR_PATH = (
    "../data/IS/ready_data/d_btc_by_year_all.pkl"
)

HYPERPARAM_DIR = (
    "../data/IS/hyperparameters"
)


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
# ======================================= TCN SEARCH SPACE ====================================================
# =============================================================================================================

def suggest_tcn_params(trial):

    channel_width = trial.suggest_categorical(
        "channel_width",
        [4, 8, 16, 32]
    )

    num_blocks = trial.suggest_categorical(
        "num_blocks",
        [4, 5, 6, 7, 8, 9]
    )

    kernel_size = trial.suggest_categorical(
        "kernel_size",
        [2, 3, 5, 7, 9, 11, 13]
    )

    dropout = trial.suggest_categorical(
        "dropout",
        [0.0, 0.10, 0.20, 0.30, 0.40]
    )

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-4,
        1e-2,
        log=True
    )

    weight_decay = trial.suggest_categorical(
        "weight_decay",
        [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [16, 32, 64, 128, 256, 512]
    )

    channels = (channel_width,) * num_blocks

    return {
        "channel_width": channel_width,
        "num_blocks": num_blocks,
        "channels": channels,
        "kernel_size": kernel_size,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
    }


# =============================================================================================================
# ======================================= CHECK TCN RECEPTIVE FIELD ===========================================
# =============================================================================================================

def valid_tcn_receptive_field(params):

    receptive_field = (
        1
        + 2
        * (params["kernel_size"] - 1)
        * (2 ** params["num_blocks"] - 1)
    )

    return receptive_field >= SEQUENCE_LENGTH


# =============================================================================================================
# ======================================= EVALUATE ONE TCN TRIAL ==============================================
# =============================================================================================================

def evaluate_tcn_trial(trial, tcn_data):

    params = suggest_tcn_params(trial)

    if not valid_tcn_receptive_field(params):
        raise optuna.TrialPruned()

    all_losses = []
    epochs_by_year = {y: [] for y in YEARS}

    resources = [1, 2, 4]

    for resource, train_fraction in zip(resources, TRAIN_FRACTIONS):

        round_losses = []

        for y in YEARS:

            fold = make_rolling_fold(
                tcn_data[y],
                train_fraction
            )

            np.random.seed(SEED)
            torch.manual_seed(SEED)

            model = SmallCausalTCN(
                n_features=tcn_data[y]["n_features"],
                channels=params["channels"],
                kernel_size=params["kernel_size"],
                dropout=params["dropout"],
            )

            val_mse, best_epoch = fit_model_fold(
                fold=fold,
                model=model,
                params=params,
                scalers=tcn_scalers,
                train=train_tcn,
                device=get_tcn_device(),
                seed=SEED,
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
# ======================================= RUN TCN TUNING ======================================================
# =============================================================================================================

def tune_tcn(tcn_data):

    sampler = optuna.samplers.TPESampler(
        seed=SEED,
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
        lambda trial: evaluate_tcn_trial(
            trial,
            tcn_data
        ),
        n_trials=N_TRIALS
    )

    return study


# =============================================================================================================
# ======================================= TCN MAIN ============================================================
# =============================================================================================================

tcn_data = prepare_model_data(
    X_by_year=X_by_year,
    outputs_by_year=outputs_by_year,
    d_btc_raw_by_year=d_btc_raw_by_year,
    df_raw_by_year=df_raw_by_year,
    model_name="TCN",
)

tcn_study = tune_tcn(tcn_data)

print("\nBEST TCN VALIDATION MSE")
print(tcn_study.best_value)

print("\nBEST TCN PARAMETERS")
print(tcn_study.best_params)

best_tcn_params = tcn_study.best_params.copy()

best_tcn_params["channels"] = (
    best_tcn_params["channel_width"],
) * best_tcn_params["num_blocks"]

print("\nTCN CHANNELS")
print(best_tcn_params["channels"])

best_tcn_params["epochs_by_year"] = {
    int(y): int(epoch)
    for y, epoch in tcn_study.best_trial.user_attrs["epochs_by_year"].items()
}

print("\nTCN EPOCHS BY YEAR")
print(best_tcn_params["epochs_by_year"])


# =============================================================================================================
# ======================================= SAVE TCN HPO RESULTS ================================================
# =============================================================================================================

os.makedirs(HYPERPARAM_DIR, exist_ok=True)

with open(f"{HYPERPARAM_DIR}/tcn_study.pkl", "wb") as f:
    pickle.dump(tcn_study, f)

with open(f"{HYPERPARAM_DIR}/tcn_best_params.pkl", "wb") as f:
    pickle.dump(best_tcn_params, f)


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
        [1, 2, 3, 4, 6, 8] # EDITED
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
        2e-5, # edited
        5e-3, # edited
        log=True
    )

    weight_decay = trial.suggest_categorical(
        "weight_decay",
        [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [32, 64, 128, 256]
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

with open(f"{HYPERPARAM_DIR}/gru_study.pkl", "wb") as f:
    pickle.dump(gru_study, f)

with open(f"{HYPERPARAM_DIR}/gru_best_params.pkl", "wb") as f:
    pickle.dump(best_gru_params, f)


# =============================================================================================================
# ======================================= TSM IMPORTS =========================================================
# =============================================================================================================

sys.modules.pop("srcB.tsm_model_mfiv", None)

from srcB.tsm_model_mfiv import (
    SmallTSMixer,
    _scalers as tsm_scalers,
    _train as train_tsm,
    _get_device as get_tsm_device,
)

TSM_SEED = 42


# =============================================================================================================
# ======================================= TSM SEARCH SPACE ====================================================
# =============================================================================================================

def suggest_tsm_params(trial):

    num_blocks = trial.suggest_categorical(
        "num_blocks",
        [1, 2, 3, 4, 5, 6]
    )

    feature_hidden = trial.suggest_categorical(
        "feature_hidden",
        [32, 64, 128, 256, 512]
    )

    dropout = trial.suggest_categorical(
        "dropout",
        [0.0, 0.10, 0.20, 0.30, 0.40]
    )

    learning_rate = trial.suggest_float(
        "learning_rate",
        2e-5,
        5e-3,
        log=True
    )

    weight_decay = trial.suggest_categorical(
        "weight_decay",
        [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [32, 64, 128, 256, 512]
    )

    return {
        "num_blocks": num_blocks,
        "feature_hidden": feature_hidden,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
    }


# =============================================================================================================
# ======================================= EVALUATE ONE TSM TRIAL ==============================================
# =============================================================================================================

def evaluate_tsm_trial(trial, tsm_data):

    params = suggest_tsm_params(trial)

    all_losses = []
    epochs_by_year = {y: [] for y in YEARS}

    resources = [1, 2, 4]

    for resource, train_fraction in zip(resources, TRAIN_FRACTIONS):

        round_losses = []

        for y in YEARS:

            fold = make_rolling_fold(
                tsm_data[y],
                train_fraction
            )

            np.random.seed(TSM_SEED)
            torch.manual_seed(TSM_SEED)

            model = SmallTSMixer(
                n_features=tsm_data[y]["n_features"],
                sequence_length=SEQUENCE_LENGTH,
                num_blocks=params["num_blocks"],
                feature_hidden=params["feature_hidden"],
                dropout=params["dropout"],
            )

            val_mse, best_epoch = fit_model_fold(
                fold=fold,
                model=model,
                params=params,
                scalers=tsm_scalers,
                train=train_tsm,
                device=get_tsm_device(),
                seed=TSM_SEED,
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
# ======================================= RUN TSM TUNING ======================================================
# =============================================================================================================

def tune_tsm(tsm_data):

    sampler = optuna.samplers.TPESampler(
        seed=TSM_SEED,
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
        lambda trial: evaluate_tsm_trial(
            trial,
            tsm_data
        ),
        n_trials=N_TRIALS
    )

    return study


# =============================================================================================================
# ======================================= TSM MAIN ============================================================
# =============================================================================================================

tsm_data = prepare_model_data(
    X_by_year=X_by_year,
    outputs_by_year=outputs_by_year,
    d_btc_raw_by_year=d_btc_raw_by_year,
    df_raw_by_year=df_raw_by_year,
    model_name="TSM",
)

tsm_study = tune_tsm(tsm_data)

print("\nBEST TSM VALIDATION MSE")
print(tsm_study.best_value)

print("\nBEST TSM PARAMETERS")
print(tsm_study.best_params)

best_tsm_params = tsm_study.best_params.copy()

best_tsm_params["epochs_by_year"] = {
    int(y): int(epoch)
    for y, epoch in tsm_study.best_trial.user_attrs["epochs_by_year"].items()
}

print("\nTSM EPOCHS BY YEAR")
print(best_tsm_params["epochs_by_year"])


# =============================================================================================================
# ======================================= SAVE TSM HPO RESULTS ================================================
# =============================================================================================================

with open(f"{HYPERPARAM_DIR}/tsm_study.pkl", "wb") as f:
    pickle.dump(tsm_study, f)

with open(f"{HYPERPARAM_DIR}/tsm_best_params.pkl", "wb") as f:
    pickle.dump(best_tsm_params, f)

# =============================================================================================================
# ======================================= ITRANSFORMER IMPORTS =================================================
# =============================================================================================================

sys.modules.pop("srcB.itr_model_mfiv", None)

from srcB.itr_model_mfiv import (
    SmallITransformer,
    _scalers as itr_scalers,
    _train as train_itr,
    _get_device as get_itr_device,
)

ITR_SEED = 42


# =============================================================================================================
# ======================================= ITRANSFORMER SEARCH SPACE ===========================================
# =============================================================================================================

def suggest_itr_params(trial):

    d_model = trial.suggest_categorical(
        "d_model",
        [16, 32, 64, 128, 256, 512]
    )

    n_heads = trial.suggest_categorical(
        "n_heads",
        [1, 2, 4, 8, 16]
    )

    e_layers = trial.suggest_categorical(
        "e_layers",
        [2, 3, 4, 6, 8, 10]
    )

    d_ff_ratio = trial.suggest_categorical(
        "d_ff_ratio",
        [2, 4, 8, 12, 16]
    )

    dropout = trial.suggest_categorical(
        "dropout",
        [0.0, 0.1, 0.2, 0.3, 0.4]
    )

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        1e-3,
        log=True
    )

    weight_decay = trial.suggest_categorical(
        "weight_decay",
        [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [32, 64, 128, 256, 512]
    )

    d_ff = d_model * d_ff_ratio

    return {
        "d_model": d_model,
        "n_heads": n_heads,
        "e_layers": e_layers,
        "d_ff_ratio": d_ff_ratio,
        "d_ff": d_ff,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
    }

# =============================================================================================================
# ======================================= EVALUATE ONE ITRANSFORMER TRIAL =====================================
# =============================================================================================================

def evaluate_itr_trial(trial, itr_data):

    params = suggest_itr_params(trial)

    all_losses = []
    epochs_by_year = {y: [] for y in YEARS}

    resources = [1, 2, 4]

    for resource, train_fraction in zip(resources, TRAIN_FRACTIONS):

        round_losses = []

        for y in YEARS:

            fold = make_rolling_fold(
                itr_data[y],
                train_fraction
            )

            np.random.seed(ITR_SEED)
            torch.manual_seed(ITR_SEED)

            model = SmallITransformer(
                n_features=itr_data[y]["n_features"],
                sequence_length=SEQUENCE_LENGTH,
                d_model=params["d_model"],
                n_heads=params["n_heads"],
                e_layers=params["e_layers"],
                d_ff=params["d_ff"],
                dropout=params["dropout"],
            )

            val_mse, best_epoch = fit_model_fold(
                fold=fold,
                model=model,
                params=params,
                scalers=itr_scalers,
                train=train_itr,
                device=get_itr_device(),
                seed=ITR_SEED,
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
# ======================================= RUN ITRANSFORMER TUNING =============================================
# =============================================================================================================

def tune_itr(itr_data):

    sampler = optuna.samplers.TPESampler(
        seed=ITR_SEED,
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
        lambda trial: evaluate_itr_trial(
            trial,
            itr_data
        ),
        n_trials=120 # previously 40 from global N_TRIALS.
    )

    return study


# =============================================================================================================
# ======================================= ITRANSFORMER MAIN ===================================================
# =============================================================================================================

itr_data = prepare_model_data(
    X_by_year=X_by_year,
    outputs_by_year=outputs_by_year,
    d_btc_raw_by_year=d_btc_raw_by_year,
    df_raw_by_year=df_raw_by_year,
    model_name="ITRANSFORMER",
)

itr_study = tune_itr(itr_data)

print("\nBEST ITRANSFORMER VALIDATION MSE")
print(itr_study.best_value)

print("\nBEST ITRANSFORMER PARAMETERS")
print(itr_study.best_params)

best_itr_params = itr_study.best_params.copy()

best_itr_params["d_ff"] = (
    best_itr_params["d_model"]
    * best_itr_params["d_ff_ratio"]
)

print("\nITR D_FF")
print(best_itr_params["d_ff"])

best_itr_params["epochs_by_year"] = {
    int(y): int(epoch)
    for y, epoch in itr_study.best_trial.user_attrs["epochs_by_year"].items()
}

print("\nITR EPOCHS BY YEAR")
print(best_itr_params["epochs_by_year"])


# =============================================================================================================
# ======================================= SAVE ITRANSFORMER HPO RESULTS =======================================
# =============================================================================================================

with open(f"{HYPERPARAM_DIR}/itr_study.pkl", "wb") as f:
    pickle.dump(itr_study, f)

with open(f"{HYPERPARAM_DIR}/itr_best_params.pkl", "wb") as f:
    pickle.dump(best_itr_params, f)


# =============================================================================================================
# ======================================= DLINEAR IMPORTS =====================================================
# =============================================================================================================

sys.modules.pop("srcB.dl_model_mfiv", None)

from srcB.dl_model_mfiv import (
    SmallDLinear,
    _scalers as dl_scalers,
    _train as train_dl,
    _get_device as get_dl_device,
)

DL_SEED = 42


# =============================================================================================================
# ======================================= DLINEAR SEARCH SPACE ================================================
# =============================================================================================================

def suggest_dl_params(trial):

    moving_avg = trial.suggest_categorical(
        "moving_avg",
        [13, 25, 49, 73]
    )

    individual = trial.suggest_categorical(
        "individual",
        [False, True]
    )

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-4,
        3e-3,
        log=True
    )

    weight_decay = trial.suggest_categorical(
        "weight_decay",
        [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [32, 64, 128, 256]
    )

    return {
        "moving_avg": moving_avg,
        "individual": individual,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
    }


# =============================================================================================================
# ======================================= EVALUATE ONE DLINEAR TRIAL ==========================================
# =============================================================================================================

def evaluate_dl_trial(trial, dl_data):

    params = suggest_dl_params(trial)

    all_losses = []
    epochs_by_year = {y: [] for y in YEARS}

    resources = [1, 2, 4]

    for resource, train_fraction in zip(resources, TRAIN_FRACTIONS):

        round_losses = []

        for y in YEARS:

            fold = make_rolling_fold(
                dl_data[y],
                train_fraction
            )

            np.random.seed(DL_SEED)
            torch.manual_seed(DL_SEED)

            model = SmallDLinear(
                n_features=dl_data[y]["n_features"],
                sequence_length=SEQUENCE_LENGTH,
                moving_avg=params["moving_avg"],
                individual=params["individual"],
            )

            val_mse, best_epoch = fit_model_fold(
                fold=fold,
                model=model,
                params=params,
                scalers=dl_scalers,
                train=train_dl,
                device=get_dl_device(),
                seed=DL_SEED,
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
# ======================================= RUN DLINEAR TUNING ==================================================
# =============================================================================================================

def tune_dl(dl_data):

    sampler = optuna.samplers.TPESampler(
        seed=DL_SEED,
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
        lambda trial: evaluate_dl_trial(
            trial,
            dl_data
        ),
        n_trials=N_TRIALS
    )

    return study


# =============================================================================================================
# ======================================= DLINEAR MAIN ========================================================
# =============================================================================================================

dl_data = prepare_model_data(
    X_by_year=X_by_year,
    outputs_by_year=outputs_by_year,
    d_btc_raw_by_year=d_btc_raw_by_year,
    df_raw_by_year=df_raw_by_year,
    model_name="DLINEAR",
)

dl_study = tune_dl(dl_data)

print("\nBEST DLINEAR VALIDATION MSE")
print(dl_study.best_value)

print("\nBEST DLINEAR PARAMETERS")
print(dl_study.best_params)

best_dl_params = dl_study.best_params.copy()

best_dl_params["epochs_by_year"] = {
    int(y): int(epoch)
    for y, epoch in dl_study.best_trial.user_attrs["epochs_by_year"].items()
}

print("\nDLINEAR EPOCHS BY YEAR")
print(best_dl_params["epochs_by_year"])


# =============================================================================================================
# ======================================= SAVE DLINEAR HPO RESULTS ============================================
# =============================================================================================================

with open(f"{HYPERPARAM_DIR}/dl_study.pkl", "wb") as f:
    pickle.dump(dl_study, f)

with open(f"{HYPERPARAM_DIR}/dl_best_params.pkl", "wb") as f:
    pickle.dump(best_dl_params, f)
