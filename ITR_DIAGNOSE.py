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


sys.path.insert(0, "/Users/shah/CODE_BOOK_4/THESIS_WORKING/THESIS_2")
sys.path.insert(0, "/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/MFIV_calibration")
os.chdir("/Users/shah/CODE_BOOK_4/THESIS_WORKING/THESIS_2")


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
    "/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/"
    "data/IS/outputs_by_year/outputs_by_year_01.pkl"
)

X_BY_YEAR_PATH = (
    "/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/"
    "data/IS/x_by_year_fut/x_by_year.pkl"
)

RFR_PATH = (
    "/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/"
    "data/IS/ready_data/d_btc_by_year_all.pkl"
)

HYPERPARAM_DIR = (
    "/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/"
    "data/IS/hyperparameters"
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


# =============================================================================================================
# ======================================= ITR HPO FOLD DIAGNOSTIC =============================================
# =============================================================================================================

with open(f"{HYPERPARAM_DIR}/itr_best_params.pkl", "rb") as f:
    best_itr_params = pickle.load(f)

ACTIONS = [
    "long_straddle",
    "short_straddle",
    "long_strangle",
    "short_strangle"
]

rows = []

for train_fraction in TRAIN_FRACTIONS:
    for y in YEARS:

        fold = make_rolling_fold(itr_data[y], train_fraction)

        X_train = fold["X_train"]
        y_train = fold["y_train"]
        X_val = fold["X_val"]
        y_val = fold["y_val"]

        x_mean, x_std, y_mean, y_std = itr_scalers(X_train, y_train)

        X_train = ((X_train - x_mean) / x_std).astype(np.float32)
        X_val_s = ((X_val - x_mean) / x_std).astype(np.float32)

        y_train_s = ((y_train - y_mean) / y_std).astype(np.float32)
        y_val_s = ((y_val - y_mean) / y_std).astype(np.float32)

        np.random.seed(ITR_SEED)
        torch.manual_seed(ITR_SEED)

        model = SmallITransformer(
            n_features=itr_data[y]["n_features"],
            sequence_length=SEQUENCE_LENGTH,
            d_model=best_itr_params["d_model"],
            n_heads=best_itr_params["n_heads"],
            e_layers=best_itr_params["e_layers"],
            d_ff=best_itr_params["d_ff"],
            dropout=best_itr_params["dropout"],
        )

        model, best_epoch, history = train_itr(
            model=model,
            X=X_train,
            y=y_train_s,
            device=get_itr_device(),
            epochs=EPOCHS,
            batch_size=best_itr_params["batch_size"],
            learning_rate=best_itr_params["learning_rate"],
            weight_decay=best_itr_params["weight_decay"],
            X_val=X_val_s,
            y_val=y_val_s,
            patience=PATIENCE,
        )

        model.eval()
        with torch.no_grad():
            pred_s = model(
                torch.from_numpy(X_val_s).to(get_itr_device())
            ).cpu().numpy()

        pred = pred_s * y_std + y_mean

        pred_action = pred.argmax(axis=1)
        true_action = y_val.argmax(axis=1)

        score = pred.max(axis=1)
        realized = y_val[np.arange(len(y_val)), pred_action]

        rows.append({
            "year": y,
            "train_fraction": train_fraction,
            "n": len(y_val),
            "best_epoch": best_epoch,
            "action_accuracy": (pred_action == true_action).mean(),
            "score_mean": score.mean(),
            "score_std": score.std(),
            "score_p10": np.quantile(score, .10),
            "score_p50": np.quantile(score, .50),
            "score_p90": np.quantile(score, .90),
            "score_target_spearman": pd.Series(score).corr(
                pd.Series(realized), method="spearman"
            ),
            "chosen_target_mean": realized.mean()
        })

itr_fold_diagnostic = pd.DataFrame(rows)
print(itr_fold_diagnostic.to_string(index=False))