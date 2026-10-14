# PRE MODELLING TESTS
# SEARCH WORDS :
# FEATURE_DATA_INTEGRITY
# FEATURE_DESCRIPTIVE_STATISTICS
# FEATURE_STATIONARITY
# FEATURE_AUTOCORRELATION
# FEATURE_CORRELATION
# FEATURE_DISTRIBUTION
# FEATURE_CROSS_YEAR_STABILITY
# LOOKBACK_168_DIAGNOSTICS

import pandas as pd
import numpy as np
import warnings
import os, sys
import pickle

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


sys.path.insert(0,"/Users/shah/CODE_BOOK_4/THESIS_WORKING/THESIS_2")
sys.path.insert(0,"/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/MFIV_calibration")

os.chdir("/Users/shah/CODE_BOOK_4/THESIS_WORKING/THESIS_2")


# =============================================================================================================
# ======================================= LOAD BASE DATA ======================================================
# =============================================================================================================

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/outputs_by_year/outputs_by_year_03.pkl","rb") as f:
    outputs_by_year = pickle.load(f)

with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/x_by_year_fut/x_by_year.pkl","rb") as f:
    X_by_year = pd.read_pickle(f)


YEARS = [2019, 2020, 2021]
QUARTERS = ["q1", "q2", "q3", "q4"]


# =============================================================================================================
# ======================================= PREPARE X_BY_YEAR ===================================================
# =============================================================================================================

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
# ======================================= BUILD COMMON MODEL FEATURES ========================================
# =============================================================================================================

MFIV_HORIZONS = [12, 24, 72, 168, 336]
RV_HORIZONS = [12, 24, 72]

features_by_year = {}

for y in YEARS:
    X_df = pd.concat([Xq_by_year[y][q] for q in QUARTERS]).sort_index().copy()
    garch_df = pd.concat([outputs_by_year[y][q]["garch"] for q in QUARTERS]).sort_index()

    X_df.index = pd.to_datetime(X_df.index, utc=True).floor("h")
    X_df = X_df.loc[~X_df.index.duplicated(keep="first")]
    X_df["garch"] = garch_df.reindex(X_df.index)

    features = pd.DataFrame(index=X_df.index)

    for H in MFIV_HORIZONS:
        features[f"MFIV_{H}"] = X_df[f"MFIV_{H}"].astype(float)

    features["garch"] = X_df["garch"].astype(float)
    features["rt"] = X_df["rt"].astype(float)

    rv = X_df["rv"].astype(float)

    for H in RV_HORIZONS:
        features[f"rv_past_{H}"] = rv.rolling(H, min_periods=H).sum().shift(1)

    for H1, H2 in zip(MFIV_HORIZONS[:-1], MFIV_HORIZONS[1:]):
        features[f"mfiv_slope_{H1}_{H2}"] = (
            features[f"MFIV_{H2}"] / H2
            - features[f"MFIV_{H1}"] / H1
        )

    features_by_year[y] = features.replace([np.inf, -np.inf], np.nan)


FEATURE_COLS = list(features_by_year[2019].columns)

features_all = pd.concat([
    features_by_year[y].assign(year=y)
    for y in YEARS
]).sort_index()

print(FEATURE_COLS)
print("n features:", len(FEATURE_COLS))


# =============================================================================================================
# ======================================= FEATURE_DATA_INTEGRITY ==============================================
# =============================================================================================================

rows = []

for y in YEARS:
    x = features_by_year[y]

    full_index = pd.date_range(
        x.index.min(),
        x.index.max(),
        freq="h",
        tz="UTC"
    )

    rows.append({
        "year": y,
        "rows": len(x),
        "duplicate_times": x.index.duplicated().sum(),
        "missing_hours": len(full_index.difference(x.index)),
        "rows_with_nan": x.isna().any(axis=1).sum(),
        "complete_rows": x.dropna().shape[0]
    })

integrity_check = pd.DataFrame(rows)

print(integrity_check.to_string(index=False))


missing_by_feature = pd.DataFrame({
    y: features_by_year[y].isna().sum()
    for y in YEARS
})

print("\nMISSING BY FEATURE")
print(missing_by_feature)


# =============================================================================================================
# ======================================= FEATURE_DESCRIPTIVE_STATISTICS =====================================
# =============================================================================================================

descriptive_stats = features_all[FEATURE_COLS].describe(
    percentiles=[.01, .05, .25, .50, .75, .95, .99]
).T

print(descriptive_stats)


# =============================================================================================================
# ======================================= FEATURE_STATIONARITY ================================================
# =============================================================================================================

rows = []

for y in YEARS:
    for col in FEATURE_COLS:
        x = features_by_year[y][col].dropna()

        adf = adfuller(x, autolag="AIC")
        kpss_test = kpss(x, regression="c", nlags="auto")

        rows.append({
            "year": y,
            "feature": col,
            "n": len(x),
            "adf_stat": adf[0],
            "adf_p": adf[1],
            "kpss_stat": kpss_test[0],
            "kpss_p": kpss_test[1],
            "adf_stationary": adf[1] < 0.05,
            "kpss_stationary": kpss_test[1] >= 0.05
        })

stationarity_check = pd.DataFrame(rows)

stationarity_check["both_support_stationary"] = (
    stationarity_check["adf_stationary"]
    & stationarity_check["kpss_stationary"]
)

print(stationarity_check.to_string(index=False))


# =============================================================================================================
# ======================================= FEATURE_AUTOCORRELATION =============================================
# =============================================================================================================

LAGS = [1, 12, 24, 72, 168]

rows = []

for y in YEARS:
    for col in FEATURE_COLS:
        x = features_by_year[y][col].dropna()

        row = {
            "year": y,
            "feature": col
        }

        for lag in LAGS:
            row[f"acf_{lag}"] = x.autocorr(lag=lag)

        rows.append(row)

autocorrelation_check = pd.DataFrame(rows)

print(autocorrelation_check.to_string(index=False))


# =============================================================================================================
# ======================================= LJUNG BOX TEST ======================================================
# =============================================================================================================

LB_LAGS = [12, 24, 72, 168]

rows = []

for y in YEARS:
    for col in FEATURE_COLS:
        x = features_by_year[y][col].dropna()

        lb = acorr_ljungbox(
            x,
            lags=LB_LAGS,
            return_df=True
        )

        for lag in LB_LAGS:
            rows.append({
                "year": y,
                "feature": col,
                "lag": lag,
                "lb_stat": lb.loc[lag, "lb_stat"],
                "lb_p": lb.loc[lag, "lb_pvalue"]
            })

ljung_box_check = pd.DataFrame(rows)

print(ljung_box_check.to_string(index=False))


# =============================================================================================================
# ======================================= FEATURE_CORRELATION =================================================
# =============================================================================================================

complete_features = features_all[FEATURE_COLS].dropna()

pearson_corr = complete_features.corr(method="pearson")
spearman_corr = complete_features.corr(method="spearman")

print("\nPEARSON CORRELATION")
print(pearson_corr)

print("\nSPEARMAN CORRELATION")
print(spearman_corr)


rows = []

for i, col1 in enumerate(FEATURE_COLS):
    for col2 in FEATURE_COLS[i + 1:]:
        corr = pearson_corr.loc[col1, col2]

        if abs(corr) >= 0.90:
            rows.append({
                "feature_1": col1,
                "feature_2": col2,
                "pearson_corr": corr
            })

high_correlation_pairs = pd.DataFrame(rows)

print("\nHIGH CORRELATION PAIRS")
print(high_correlation_pairs.to_string(index=False))


# =============================================================================================================
# ======================================= FEATURE_DISTRIBUTION ================================================
# =============================================================================================================

rows = []

for col in FEATURE_COLS:
    x = complete_features[col]
    jb = jarque_bera(x)

    rows.append({
        "feature": col,
        "n": len(x),
        "mean": x.mean(),
        "std": x.std(),
        "skew": x.skew(),
        "excess_kurtosis": x.kurt(),
        "jb_stat": jb.statistic,
        "jb_p": jb.pvalue
    })

distribution_check = pd.DataFrame(rows)

print(distribution_check.to_string(index=False))


# =============================================================================================================
# ======================================= FEATURE_CROSS_YEAR_STABILITY =======================================
# =============================================================================================================

rows = []

for y in YEARS:
    for col in FEATURE_COLS:
        x = features_by_year[y][col].dropna()

        rows.append({
            "year": y,
            "feature": col,
            "n": len(x),
            "mean": x.mean(),
            "std": x.std(),
            "median": x.median(),
            "q25": x.quantile(.25),
            "q75": x.quantile(.75)
        })

cross_year_check = pd.DataFrame(rows)

print(cross_year_check.to_string(index=False))


mean_by_year = cross_year_check.pivot(
    index="feature",
    columns="year",
    values="mean"
)

std_by_year = cross_year_check.pivot(
    index="feature",
    columns="year",
    values="std"
)

print("\nMEAN BY YEAR")
print(mean_by_year)

print("\nSTD BY YEAR")
print(std_by_year)


# =============================================================================================================
# ======================================= LOOKBACK_168_DIAGNOSTICS ============================================
# =============================================================================================================

SEQUENCE_LENGTH = 168

rows = []

for y in YEARS:
    x = features_by_year[y]
    valid_windows = 0

    for i in range(SEQUENCE_LENGTH - 1, len(x)):
        seq = x.iloc[i - SEQUENCE_LENGTH + 1:i + 1]

        if seq.index[-1] - seq.index[0] != pd.Timedelta(hours=SEQUENCE_LENGTH - 1):
            continue

        if seq.isna().any().any():
            continue

        valid_windows += 1

    possible_windows = max(len(x) - SEQUENCE_LENGTH + 1, 0)

    rows.append({
        "year": y,
        "possible_windows": possible_windows,
        "valid_168h_windows": valid_windows,
        "valid_fraction": valid_windows / possible_windows if possible_windows else np.nan
    })

lookback_window_check = pd.DataFrame(rows)

print(lookback_window_check.to_string(index=False))


lookback_dependence = autocorrelation_check[
    ["year", "feature", "acf_24", "acf_72", "acf_168"]
].copy()

print("\nLOOKBACK DEPENDENCE")
print(lookback_dependence.to_string(index=False))


adjacent_window_overlap = (SEQUENCE_LENGTH - 1) / SEQUENCE_LENGTH

print("\nADJACENT 168-HOUR WINDOW OVERLAP:", adjacent_window_overlap)


# =============================================================================================================
# ======================================= CHECKPOINT 1 ========================================================
# =============================================================================================================
# WE NOW HAVE :
# integrity_check
# missing_by_feature
# descriptive_stats
# stationarity_check
# autocorrelation_check
# ljung_box_check
# pearson_corr
# spearman_corr
# high_correlation_pairs
# distribution_check
# cross_year_check
# mean_by_year
# std_by_year
# lookback_window_check
# lookback_dependence


# =============================================================================================================
# ======================================= FEATURE_OUTLIER_DIAGNOSTICS =========================================
# =============================================================================================================

rows = []

for y in YEARS:
    for col in FEATURE_COLS:
        x = features_by_year[y][col].dropna()

        q25 = x.quantile(.25)
        q75 = x.quantile(.75)
        robust_std = (q75 - q25) / 1.349
        z = (x - x.mean()) / x.std()

        rows.append({
            "year": y,
            "feature": col,
            "median": x.median(),
            "q95": x.quantile(.95),
            "q99": x.quantile(.99),
            "q999": x.quantile(.999),
            "min": x.min(),
            "max": x.max(),
            "std": x.std(),
            "robust_std": robust_std,
            "std_to_robust": x.std() / robust_std if robust_std > 0 else np.nan,
            "max_abs_z": z.abs().max()
        })

outlier_check = pd.DataFrame(rows)

print(outlier_check.to_string(index=False))


# =============================================================================================================
# ======================================= GARCH EXTREME OBSERVATIONS ==========================================
# =============================================================================================================

garch_extremes = []

for y in YEARS:
    x = features_by_year[y].copy()

    top = x.nlargest(15, "garch")[
        ["garch", "rt", "rv_past_12", "rv_past_24", "rv_past_72",
         "MFIV_12", "MFIV_24", "MFIV_72"]
    ].copy()

    top["year"] = y
    garch_extremes.append(top)

garch_extremes = pd.concat(garch_extremes).sort_values("garch", ascending=False)

print(garch_extremes.to_string())

# The first thing I would separate is a probable GARCH data/estimation anomaly from the broader, genuine heavy-tailed structure of volatility features and targets.
# The maximum is about 66× the 99.9th percentile and its standardized magnitude is about 93 standard deviations. Your standard deviation is consequently 226 times the robust scale estimate in 2020.
# =============================================================================================================
# ======================================= TARGET_DISTRIBUTION_DIAGNOSTICS =====================================
# =============================================================================================================

from pathlib import Path
from srcB.tcn_model_mfiv import build_forward_targets, make_training_sequences, TARGET_COLS


DATA_DIR = Path("../data/rawData")

d_btc_raw_by_year = {}
df_raw_by_year = {}

for y in YEARS:
    d_btc_raw_by_year[y] = pd.read_parquet(DATA_DIR/f"btc_underlying_minutely_{y}.parquet")
    df_raw_by_year[y] = pd.read_parquet(DATA_DIR/f"derivative_data_{y}.parquet")

for y in YEARS:
    d_btc_raw_by_year[y] = d_btc_raw_by_year[y].set_index(
        pd.to_datetime(d_btc_raw_by_year[y]["date_time"])
    )
    df_raw_by_year[y] = df_raw_by_year[y].set_index(
        pd.to_datetime(df_raw_by_year[y]["date_time"])
    )


with open("/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/ready_data/d_btc_by_year_all.pkl","rb") as f:
    d_btc_by_year = pickle.load(f)

for y in YEARS:
    d_btc_raw_by_year[y]["rfr"] = d_btc_by_year[y]["rfr"].reindex(
        d_btc_raw_by_year[y].index,
        method="ffill"
    )


targets_by_year = {}
model_targets_by_year = {}

for y in YEARS:
    targets = build_forward_targets(
        option_df=df_raw_by_year[y],
        time_df=d_btc_raw_by_year[y],
        times=features_by_year[y].dropna().index,
        forward_hours=24,
        delta_target=.20
    )

    _, y_array, idx = make_training_sequences(
        features_by_year[y],
        targets,
        sequence_length=168
    )

    targets_by_year[y] = targets

    model_targets_by_year[y] = pd.DataFrame(
        y_array,
        index=idx,
        columns=TARGET_COLS
    )


rows = []

for y in YEARS:
    for col in TARGET_COLS:
        x = model_targets_by_year[y][col].dropna()

        rows.append({
            "year": y,
            "target": col,
            "n": len(x),
            "mean": x.mean(),
            "std": x.std(),
            "min": x.min(),
            "q01": x.quantile(.01),
            "q05": x.quantile(.05),
            "q25": x.quantile(.25),
            "median": x.median(),
            "q75": x.quantile(.75),
            "q95": x.quantile(.95),
            "q99": x.quantile(.99),
            "max": x.max(),
            "skew": x.skew(),
            "excess_kurtosis": x.kurt()
        })

target_distribution_check = pd.DataFrame(rows)
print(target_distribution_check.to_string(index=False))
target_distribution_check.plot()

# I thought the standardScaler was enough but
# now I strongly favour log + StandardScaler for the positive variance features
# | Features    | Transformation                                      |
# | ----------- | --------------------------------------------------  |
# | MFIV levels |  log  -> StandardScaler                             |
# | GARCH       | investigate bad point, then log -> StandardScaler   |
# | Past RV     |  log -> StandardScaler                              |
# | rt          | StandardScaler                                      |
# | MFIV slopes | StandardScaler                                      |


# Regarding the scaling on y, currently each action has its own y_std 
# so the multi-output MSE is effectively weighting each original PnL target approximately by 1/sigma_sq_action. 
# So targets with smaller PnL variance receive relatively more weight in the standardized MSE
# Standardization therefore intentionally makes a unit_std_dev error in a strangle comparable with a unit-std-dev error in a straddle.
# So all four actions are learned adequetly despite their inherent PnL dispersion.


