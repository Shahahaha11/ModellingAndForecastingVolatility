# MFIV variant of dl_model_no_mfiv.py.
# Same architecture, targets, training and prediction; MFIV enters through build_dl_features().
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from torch.utils.data import DataLoader, TensorDataset


ACTIONS = [
    "long_straddle",
    "short_straddle",
    "long_strangle",
    "short_strangle",
]
TARGET_COLS = [f"target_{x}" for x in ACTIONS]

# ==================================================================================================
# FEATURES
# ==================================================================================================

def build_dl_features(
    X_df,
    mfiv_horizons=(12, 24, 72, 168, 336),
    rv_horizons=(12, 24, 72),
):
    X_df = X_df.copy().sort_index()
    X_df.index = pd.to_datetime(X_df.index, utc=True).floor("h")
    X_df = X_df.loc[~X_df.index.duplicated(keep="first")]

    features = pd.DataFrame(index=X_df.index)

    for H in mfiv_horizons:
        features[f"MFIV_{H}"] = X_df[f"MFIV_{H}"].astype(float)

    features["garch"] = X_df["garch"].astype(float)
    features["rt"] = X_df["rt"].astype(float)

    rv = X_df["rv"].astype(float)

    for H in rv_horizons:
        features[f"rv_past_{H}"] = rv.rolling(H, min_periods=H).sum().shift(1)

    # MFIV_H is cumulative H-hour variance, so compare the curve
    # in variance-per-hour units.
    for H1, H2 in zip(mfiv_horizons[:-1], mfiv_horizons[1:]):
        features[f"mfiv_slope_{H1}_{H2}"] = (
            features[f"MFIV_{H2}"] / H2
            - features[f"MFIV_{H1}"] / H1
        )

    positive_cols = [
        "MFIV_12",
        "MFIV_24",
        "MFIV_72",
        "MFIV_168",
        "MFIV_336",
        "garch",
        "rv_past_12",
        "rv_past_24",
        "rv_past_72",
    ]

    for col in positive_cols:
        features[col] = np.log(features[col].where(features[col] > 0))

    return features.replace([np.inf, -np.inf], np.nan)

# ==================================================================================================
# 24-HOUR COUNTERFACTUAL OPTION TARGETS
# ==================================================================================================

def _prepare_option_data(option_df):
    option = option_df.copy().reset_index(drop=True)
    option["date_time"] = pd.to_datetime(option["date_time"], utc=True).dt.ceil("h")

    if "strike" not in option.columns:
        option["strike"] = option["instrument_name"].str.split("-").str[2].astype(float)

    if "type" not in option.columns:
        option["type"] = option["instrument_name"].str.split("-").str[3]

    if "maturity" not in option.columns:
        option["maturity"] = pd.to_datetime(
            option["instrument_name"].str.split("-").str[1],
            format="%d%b%y",
            utc=True,
        ).dt.floor("h")
    else:
        option["maturity"] = pd.to_datetime(
            option["maturity"],
            utc=True
        ).dt.floor("h")

    option["price_per_contract"] = (
        option["price"].astype(float)
        * option["index_price"].astype(float)
    )
    option["iv"] = option["iv"].astype(float)
    option["amount"] = option["amount"].astype(float)
    return option.sort_values("date_time", kind="stable")


def _prepare_time_data(time_df):
    time_df = time_df.copy()
    time_df.index = pd.to_datetime(time_df.index, utc=True).ceil("h")
    time_df = time_df.loc[~time_df.index.duplicated(keep="last")].sort_index()
    close_col = "close" if "close" in time_df.columns else "Close"
    return time_df, close_col


def _delta(now, spot, strike, rfr, iv, expiry, right):
    sigma = iv / 100.0
    tau = (expiry - now) / pd.Timedelta(days=365)

    if not np.isfinite(sigma) or sigma <= 0 or tau <= 0:
        return np.nan

    d1 = (
        np.log(spot / strike)
        + (rfr + 0.5 * sigma * sigma) * tau
    ) / (sigma * np.sqrt(tau))

    return norm.cdf(d1) if right == "C" else norm.cdf(d1) - 1.0


def _fee(option_price_btc, spot):
    return min(0.0003, 0.125 * option_price_btc) * spot


def _build_quote_groups(option):
    return {
        key: group.set_index("date_time").sort_index()
        for key, group in option.groupby(["instrument_name", "direction"], sort=False)
    }


def _next_quote(
    quote_groups,
    target_time,
    symbol,
    direction,
    expiry,
    quantity=0.1,
):
    quotes = quote_groups.get((symbol, direction))

    if quotes is None or quotes.empty:
        return [], quantity

    quotes = quotes[
        (quotes.index >= target_time)
        & (quotes.index < expiry)
    ]

    remaining = float(quantity)
    fills = []

    for quote_time, row in quotes.iterrows():
        available = float(row["amount"])

        if not np.isfinite(available) or available <= 0:
            continue

        fill_qty = min(remaining, available)

        fills.append({
            "time": quote_time,
            "row": row,
            "quantity": fill_qty,
        })

        remaining -= fill_qty

        if remaining <= 1e-12:
            remaining = 0.0
            break

    return fills, remaining


def _leg_pnl(
    quote_groups, time_df, close_col,
    now, target_exit_time, entry_row, right, direction,
):
    symbol = entry_row["instrument_name"]
    strike = float(entry_row["strike"])
    expiry = pd.Timestamp(entry_row["maturity"])

    quantity = 0.1

    spot0 = float(time_df.loc[now, close_col])
    entry_price = float(entry_row["price_per_contract"])
    entry_fee = _fee(float(entry_row["price"]), spot0)

    if direction == "buy":
        pnl = (-entry_price - entry_fee) * quantity
    else:
        pnl = (entry_price - entry_fee) * quantity

    # Expiry occurs before the intended holding horizon.
    if expiry <= target_exit_time:
        if expiry not in time_df.index:
            return np.nan, pd.NaT

        spot1 = float(time_df.loc[expiry, close_col])
        payoff = (
            max(spot1 - strike, 0.0)
            if right == "C"
            else max(strike - spot1, 0.0)
        )

        delivery_fee = (
            min(0.00015 * spot1, 0.125 * payoff)
            if payoff > 0
            else 0.0
        )

        if direction == "buy":
            pnl += (payoff - delivery_fee) * quantity
        else:
            pnl += (-payoff - delivery_fee) * quantity

        return pnl, expiry

    opposite = "sell" if direction == "buy" else "buy"

    fills, remaining = _next_quote(
        quote_groups,
        target_exit_time,
        symbol,
        opposite,
        expiry,
        quantity=quantity,
    )

    last_exit_time = None

    for fill in fills:
        exit_time = fill["time"]
        exit_row = fill["row"]
        fill_qty = fill["quantity"]

        if exit_time not in time_df.index:
            return np.nan, pd.NaT

        spot1 = float(time_df.loc[exit_time, close_col])
        exit_price = float(exit_row["price_per_contract"])
        exit_fee = _fee(float(exit_row["price"]), spot1)

        if direction == "buy":
            pnl += (exit_price - exit_fee) * fill_qty
        else:
            pnl += (-exit_price - exit_fee) * fill_qty

        last_exit_time = exit_time

    if remaining == 0:
        return pnl, last_exit_time

    if expiry not in time_df.index:
        return np.nan, pd.NaT

    spot1 = float(time_df.loc[expiry, close_col])
    payoff = (
        max(spot1 - strike, 0.0)
        if right == "C"
        else max(strike - spot1, 0.0)
    )

    delivery_fee = (
        min(0.00015 * spot1, 0.125 * payoff)
        if payoff > 0
        else 0.0
    )

    if direction == "buy":
        pnl += (payoff - delivery_fee) * remaining
    else:
        pnl += (-payoff - delivery_fee) * remaining

    return pnl, expiry

def _entry_contracts(snapshot, now, spot, rfr, exit_time, delta_target, action):
    direction = "buy" if action.startswith("long_") else "sell"
    structure = "straddle" if action.endswith("straddle") else "strangle"
    contracts = snapshot[
        (snapshot["direction"] == direction)
        & (snapshot["amount"] >= 0.1)
    ].drop_duplicates("instrument_name")

    expiries = sorted(
        x for x in contracts["maturity"].unique()
        if pd.Timestamp(x) >= exit_time
    )
    for expiry in expiries:
        expiry = pd.Timestamp(expiry)
        exp = contracts[contracts["maturity"] == expiry]
        calls = exp[exp["type"] == "C"]
        puts = exp[exp["type"] == "P"]
        
        if calls.empty or puts.empty:
            continue
        
        if structure == "straddle":
            common_strikes = sorted(set(calls["strike"]) & set(puts["strike"]))
            if not common_strikes:
                continue
            atm_strike = min(common_strikes, key=lambda K: abs(K - spot))
            call = calls[calls["strike"] == atm_strike].iloc[0]
            put = puts[puts["strike"] == atm_strike].iloc[0]
            return {
                "expiry": expiry,
                "call": call,
                "put": put,
            }

        otm_calls = calls[calls["strike"] > spot]
        otm_puts = puts[puts["strike"] < spot]
        
        if otm_calls.empty or otm_puts.empty:
            continue
        
        call_choices = []
        for _, row in otm_calls.iterrows():
            d = _delta(
                now, spot, float(row["strike"]), rfr,
                float(row["iv"]), expiry, "C"
            )
            if np.isfinite(d):
                call_choices.append((abs(d - delta_target), row))

        put_choices = []
        for _, row in otm_puts.iterrows():
            d = _delta(
                now, spot, float(row["strike"]), rfr,
                float(row["iv"]), expiry, "P"
            )
            if np.isfinite(d):
                put_choices.append((abs(abs(d) - delta_target), row))

        if not call_choices or not put_choices:
            continue
        
        return {
            "expiry": expiry,
            "call": min(call_choices, key=lambda x: x[0])[1],
            "put": min(put_choices, key=lambda x: x[0])[1],
        }
    return None


# ==================================================================================================
# TARGET DIAGNOSTICS
# ==================================================================================================

def diagnose_forward_targets(
    option_df,
    time_df,
    times=None,
    forward_hours=24,
    delta_target=0.20,
):
    """Count entry and realized target availability for each candidate action."""
    option = _prepare_option_data(option_df)
    time_df, close_col = _prepare_time_data(time_df)

    grouped = option.groupby("date_time", sort=False)
    quote_groups = _build_quote_groups(option)

    if times is None:
        times = time_df.index
    else:
        times = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).floor("h")

    times = times.intersection(time_df.index).sort_values()
    horizon = pd.Timedelta(hours=forward_hours)

    counts = {
        "candidate_times": 0,
        "missing_exit_spot": 0,
        "missing_entry_snapshot": 0,
    }

    for action in ACTIONS:
        counts[f"{action}_entry_valid"] = 0
        counts[f"{action}_valid"] = 0

    counts["all_four_entry_valid"] = 0
    counts["all_four_valid"] = 0

    for now in times:
        counts["candidate_times"] += 1
        target_exit_time = now + horizon

        if target_exit_time not in time_df.index:
            counts["missing_exit_spot"] += 1
            continue

        try:
            snapshot = grouped.get_group(now)
        except KeyError:
            counts["missing_entry_snapshot"] += 1
            continue

        spot = float(time_df.loc[now, close_col])
        rfr = float(time_df.loc[now, "rfr"])

        entry_valid = {}
        target_valid = {}

        for action in ACTIONS:
            selected = _entry_contracts(
                snapshot, now, spot, rfr,
                target_exit_time, delta_target, action
            )

            entry_valid[action] = selected is not None
            if not entry_valid[action]:
                target_valid[action] = False
                continue

            counts[f"{action}_entry_valid"] += 1

            direction = "buy" if action.startswith("long_") else "sell"
            call_pnl, _ = _leg_pnl(
                quote_groups, time_df, close_col,
                now, target_exit_time, selected["call"], "C", direction,
            )
            put_pnl, _ = _leg_pnl(
                quote_groups, time_df, close_col,
                now, target_exit_time, selected["put"], "P", direction,
            )

            target_valid[action] = np.isfinite(call_pnl) and np.isfinite(put_pnl)
            if target_valid[action]:
                counts[f"{action}_valid"] += 1

        if all(entry_valid.get(action, False) for action in ACTIONS):
            counts["all_four_entry_valid"] += 1

        if all(target_valid.get(action, False) for action in ACTIONS):
            counts["all_four_valid"] += 1

    return pd.Series(counts, name="count")


def build_forward_targets(
    option_df,
    time_df,
    times=None,
    forward_hours=24,
    delta_target=0.20,
):
    """
    Four action targets at t:
        ATM long straddle, ATM short straddle,
        20-delta long strangle, 20-delta short strangle.

    Each action uses the nearest executable common expiry with at least
    forward_hours remaining. The intended exit is t + forward_hours; if the
    opposite-direction trade is unavailable then, liquidation is delayed until
    the first executable hour, exactly as in Market.liquidate(), or expiry.

    Target = realized PnL / entry spot.
    """
    option = _prepare_option_data(option_df)
    time_df, close_col = _prepare_time_data(time_df)

    grouped = option.groupby("date_time", sort=False)
    quote_groups = _build_quote_groups(option)

    if times is None:
        times = time_df.index
    else:
        times = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).floor("h")

    rows = []
    horizon = pd.Timedelta(hours=forward_hours)

    for now in times.intersection(time_df.index).sort_values():
        target_exit_time = now + horizon
        if target_exit_time not in time_df.index:
            continue

        try:
            snapshot = grouped.get_group(now)
        except KeyError:
            continue

        spot = float(time_df.loc[now, close_col])
        rfr = float(time_df.loc[now, "rfr"])

        row = {"time": now}
        valid = True

        for action in ACTIONS:
            selected = _entry_contracts(
                snapshot, now, spot, rfr,
                target_exit_time, delta_target, action
            )

            if selected is None:
                valid = False
                break

            direction = "buy" if action.startswith("long_") else "sell"

            call_pnl, call_exit = _leg_pnl(
                quote_groups, time_df, close_col,
                now, target_exit_time, selected["call"], "C", direction,
            )
            put_pnl, put_exit = _leg_pnl(
                quote_groups, time_df, close_col,
                now, target_exit_time, selected["put"], "P", direction,
            )

            pnl = call_pnl + put_pnl
            if not np.isfinite(pnl):
                valid = False
                break

            row[f"pnl_{action}"] = pnl
            row[f"target_{action}"] = pnl / spot
            row[f"{action}_expiry"] = selected["expiry"]
            row[f"{action}_call_strike"] = float(selected["call"]["strike"])
            row[f"{action}_put_strike"] = float(selected["put"]["strike"])
            row[f"{action}_exit_time"] = max(call_exit, put_exit)

        if valid:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("time").sort_index()


# ==================================================================================================
# DLINEAR
# ==================================================================================================

class MovingAverage(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.transpose(1, 2))
        return x.transpose(1, 2)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_average = MovingAverage(kernel_size)

    def forward(self, x):
        trend = self.moving_average(x)
        seasonal = x - trend
        return seasonal, trend


class SmallDLinear(nn.Module):
    def __init__(
        self,
        n_features,
        sequence_length=168,
        moving_avg=25,
        individual=False,
    ):
        super().__init__()

        self.n_features = n_features
        self.individual = individual
        self.decomposition = SeriesDecomposition(moving_avg)

        if individual:
            self.seasonal = nn.ModuleList([
                nn.Linear(sequence_length, 1)
                for _ in range(n_features)
            ])
            self.trend = nn.ModuleList([
                nn.Linear(sequence_length, 1)
                for _ in range(n_features)
            ])
        else:
            self.seasonal = nn.Linear(sequence_length, 1)
            self.trend = nn.Linear(sequence_length, 1)

        self.output = nn.Linear(n_features, len(ACTIONS))

    def forward(self, x):
        seasonal, trend = self.decomposition(x)
        seasonal = seasonal.transpose(1, 2)
        trend = trend.transpose(1, 2)

        if self.individual:
            seasonal = torch.cat([
                self.seasonal[i](seasonal[:, i, :])
                for i in range(self.n_features)
            ], dim=1)
            trend = torch.cat([
                self.trend[i](trend[:, i, :])
                for i in range(self.n_features)
            ], dim=1)
        else:
            seasonal = self.seasonal(seasonal).squeeze(-1)
            trend = self.trend(trend).squeeze(-1)

        return self.output(seasonal + trend)


# ==================================================================================================
# DATASET
# ==================================================================================================

def make_training_sequences(features, targets, sequence_length=168):
    X, y, idx = [], [], []
    feature_index = features.index

    for now in targets.index.intersection(feature_index):
        i = feature_index.get_loc(now)
        if i < sequence_length - 1:
            continue

        seq = features.iloc[i - sequence_length + 1:i + 1]
        if seq.index[-1] - seq.index[0] != pd.Timedelta(hours=sequence_length - 1):
            continue
        if seq.isna().any().any() or targets.loc[now, TARGET_COLS].isna().any():
            continue

        X.append(seq.to_numpy(np.float32))
        y.append(targets.loc[now, TARGET_COLS].to_numpy(np.float32))
        idx.append(now)

    if not X:
        raise ValueError("No valid DLinear training sequences")

    return np.asarray(X), np.asarray(y), pd.DatetimeIndex(idx)


def make_prediction_sequences(features, sequence_length=168):
    X, idx = [], []

    for i in range(sequence_length - 1, len(features)):
        seq = features.iloc[i - sequence_length + 1:i + 1]
        if seq.index[-1] - seq.index[0] != pd.Timedelta(hours=sequence_length - 1):
            continue
        if seq.isna().any().any():
            continue

        X.append(seq.to_numpy(np.float32))
        idx.append(seq.index[-1])

    return np.asarray(X), pd.DatetimeIndex(idx)


# ==================================================================================================
# TRAIN / PREDICT
# ==================================================================================================

def _get_device(device=None):
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _scalers(X, y):
    x_mean = X.mean((0, 1), keepdims=True).astype(np.float32)
    x_std = X.std((0, 1), keepdims=True).astype(np.float32)
    x_std[x_std < 1e-8] = 1.0

    y_mean = y.mean(0, keepdims=True).astype(np.float32)
    y_std = y.std(0, keepdims=True).astype(np.float32)
    y_std[y_std < 1e-8] = 1.0

    return x_mean, x_std, y_mean, y_std


def _train(
    model, X, y, device,
    epochs, batch_size, learning_rate, weight_decay,
    X_val=None, y_val=None, patience=6,
):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.MSELoss()
    model = model.to(device)

    best_state = None
    best_epoch = epochs
    best_val = np.inf
    stale = 0
    history = []

    for epoch in range(epochs):
        model.train()
        losses = []

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        row = {"epoch": epoch + 1, "train_loss": np.mean(losses)}

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                xb = torch.from_numpy(X_val).to(device)
                yb = torch.from_numpy(y_val).to(device)
                val_loss = float(loss_fn(model(xb), yb).item())

            row["val_loss"] = val_loss

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1

            if stale >= patience:
                history.append(row)
                break

        history.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_epoch, pd.DataFrame(history)


def fit_dl(
    X,
    y,
    n_features,
    moving_avg=25,
    individual=False,
    epochs=40,
    batch_size=128,
    learning_rate=1e-3,
    weight_decay=1e-4,
    seed=42,
    device=None,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    x_mean, x_std, y_mean, y_std = _scalers(X, y)

    X = ((X - x_mean) / x_std).astype(np.float32)
    y = ((y - y_mean) / y_std).astype(np.float32)

    run_device = _get_device(device)
    sequence_length = X.shape[1]

    model = SmallDLinear(
        n_features=n_features,
        sequence_length=sequence_length,
        moving_avg=moving_avg,
        individual=individual,
    )

    model, _, history = _train(
        model,
        X,
        y,
        run_device,
        int(epochs),
        batch_size,
        learning_rate,
        weight_decay,
    )

    return {
        "model": model,
        "device": str(run_device),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "fit_epochs": int(epochs),
        "history": history,
    }

def predict_dl(bundle, X_df):
    features = build_dl_features(X_df)[bundle["feature_cols"]]
    X, idx = make_prediction_sequences(features, bundle["sequence_length"])

    if len(X) == 0:
        return pd.DataFrame(index=idx)

    X = ((X - bundle["x_mean"]) / bundle["x_std"]).astype(np.float32)

    model = bundle["model"].to(bundle["device"])
    model.eval()

    pred = []
    with torch.no_grad():
        for start in range(0, len(X), 512):
            xb = torch.from_numpy(X[start:start + 512]).to(bundle["device"])
            pred.append(model(xb).cpu().numpy())

    pred = np.vstack(pred) * bundle["y_std"] + bundle["y_mean"]

    out = pd.DataFrame(
        pred,
        index=idx,
        columns=[f"dl_{a}" for a in ACTIONS],
    )

    score_cols = list(out.columns)
    out["dl_best_score"] = out[score_cols].max(axis=1)
    best = out[score_cols].idxmax(axis=1).str.replace("dl_", "", regex=False)
    out["dl_action"] = np.where(out["dl_best_score"] > 0, best, "no_trade")

    return out


# ==================================================================================================
# FRONT END FOR analysis.py
# ==================================================================================================

def dl_mfiv(
    X_df,
    option_df,
    time_df,
    sequence_length=168,
    forward_hours=24,
    delta_target=0.20,
    moving_avg=25,
    individual=False,
    epochs=40,
    batch_size=128,
    learning_rate=1e-3,
    weight_decay=1e-4,
    seed=42,
    device=None,
):
    features = build_dl_features(X_df)

    targets = build_forward_targets(
        option_df=option_df,
        time_df=time_df,
        times=features.dropna().index,
        forward_hours=forward_hours,
        delta_target=delta_target,
    )

    X, y, _ = make_training_sequences(
        features,
        targets,
        sequence_length=sequence_length,
    )

    bundle = fit_dl(
        X=X,
        y=y,
        n_features=features.shape[1],
        moving_avg=moving_avg,
        individual=individual,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
    )

    bundle["feature_cols"] = list(features.columns)
    bundle["sequence_length"] = sequence_length
    bundle["forward_hours"] = forward_hours
    bundle["delta_target"] = delta_target
    bundle["moving_avg"] = moving_avg
    bundle["individual"] = individual

    fitted = predict_dl(bundle, X_df)

    return fitted, bundle, targets
