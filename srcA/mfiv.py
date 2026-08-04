import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from tqdm import tqdm

# This class computes a daily model-free implied volatility from option data by integrating call prices across strikes.
# It first filters data, keeps only call options, and groups observations by day.
# For each day, it selects the option expiry closest to a target maturity and one representative intraday timestamp.
# It then builds the forward price, integrates adjusted call prices over strike to obtain implied variance, and converts that into an implied volatility for the day.

# That one representative time stamp is not a good measure in my opinion. 
# I need to compare three methods of aggregating options per time stamp.
# The main question : What frequnecy of data to talk about when the options are time stamped randomly within that frequency.
# Hourly frequency should be alright.
class MFIVFiniteGridDailyNearestExpiry:
    def __init__(self, day_count: float = 365.0, q: float = 0.0):
        self.day_count = float(day_count)
        self.q = float(q)

    @staticmethod
    def _is_call(type_series: pd.Series) -> pd.Series:
        t = type_series.astype(str).str.lower().str.strip()
        return t.isin(["c"]) | t.str.contains("call", na=False)

    def compute_daily(
        self,
        df: pd.DataFrame,
        r: float,
        min_strikes: int = 5,
        target_days: float = 30.0,
        time_rule: str = "last",     # "last", "first", "median"
        time_window: str | None = None,  # e.g. "15:30-16:00" in UTC, or None
    ) -> pd.DataFrame:
        x = df.copy()
        x = x[self._is_call(x["type"])]

        x["date_time"] = pd.to_datetime(x["date_time"], errors="coerce", utc=True).dt.floor("min")
        x["maturity"]  = pd.to_datetime(x["maturity"],  errors="coerce", utc=True)

        x["strike"] = pd.to_numeric(x["strike"], errors="coerce")
        x["option_price"] = pd.to_numeric(x["option_price"], errors="coerce")
        x["index_price"] = pd.to_numeric(x["index_price"], errors="coerce")

        x = x.dropna(subset=["date_time","maturity","strike","option_price","index_price"])
        x = x[(x["strike"] > 0) & (x["maturity"] > x["date_time"])]

        if x.empty:
            return pd.DataFrame(columns=[
                "date","asof_time","expiry_used","T_years","T_days","T_minutes",
                "S0","F0","Kmin","Kmax","mfiv_var","mfiv_vol","n_strikes"
            ])

        r = float(r)
        out = []

        # day bucket (UTC day); you can switch to local exchange day if needed
        x["date"] = x["date_time"].dt.floor("D")

        for d, gday in x.groupby("date", sort=False):
            g = gday

            # optional: restrict to an intraday window (UTC)
            if time_window is not None:
                start_s, end_s = time_window.split("-")
                start_t = pd.to_datetime(f"{d.date()} {start_s}", utc=True)
                end_t   = pd.to_datetime(f"{d.date()} {end_s}",   utc=True)
                g = g[(g["date_time"] >= start_t) & (g["date_time"] <= end_t)]
                if g.empty:
                    continue

            # pick expiry closest to target horizon using *available rows that day*
            # compute days-to-expiry for each row; then choose expiry that minimizes |T_days - target_days|
            T_days_row = (g["maturity"] - g["date_time"]).dt.total_seconds() / (24.0 * 3600.0)
            g = g.assign(_T_days=T_days_row)
            g = g[g["_T_days"] > 0]
            if g.empty:
                continue

            # choose expiry by day-level criterion
            expiry_scores = (g.groupby("maturity")["_T_days"]
                               .median()
                               .sub(target_days).abs())
            expiry_used = expiry_scores.idxmin()
            h = g[g["maturity"] == expiry_used]
            if h.empty:
                continue

            # choose one "as-of" timestamp inside the day+expiry slice
            if time_rule == "last":
                asof_time = h["date_time"].max()
            elif time_rule == "first":
                asof_time = h["date_time"].min()
            elif time_rule == "median":
                asof_time = h["date_time"].sort_values().iloc[len(h)//2]
            else:
                raise ValueError("time_rule must be one of: 'last','first','median'")

            ht = h[h["date_time"] == asof_time]
            if ht.empty:
                continue

            # require distinct strikes at that timestamp; average duplicates
            if ht["strike"].nunique() < min_strikes:
                continue

            hh = (ht.groupby("strike", as_index=False)["option_price"].mean()
                    .sort_values("strike"))

            T_years = (expiry_used - asof_time).total_seconds() / (24 * 3600) / self.day_count
            if T_years <= 0:
                continue

            S0 = float(ht["index_price"].iloc[0])
            disc = np.exp(-r * T_years)
            F0 = S0 * np.exp((r - self.q) * T_years)

            K = hh["strike"].to_numpy(float)
            C = hh["option_price"].to_numpy(float)

            C_F = C / disc
            intrinsic_fwd = np.maximum(0.0, F0 - K)
            integrand = (C_F - intrinsic_fwd) / (K ** 2)

            mfiv_var = max(0.0, 2.0 * np.trapezoid(integrand, K))
            mfiv_vol = np.sqrt(mfiv_var / T_years)

            T_minutes = (expiry_used - asof_time).total_seconds() / 60.0
            T_days    = T_minutes / (24.0 * 60.0)

            out.append({
                "date": d,
                "asof_time": asof_time,
                "expiry_used": expiry_used,
                "T_years": T_years,
                "T_days": T_days,
                "T_minutes": T_minutes,
                "S0": S0,
                "F0": F0,
                "Kmin": float(K.min()),
                "Kmax": float(K.max()),
                "mfiv_var": mfiv_var,
                "mfiv_vol": mfiv_vol,
                "n_strikes": int(len(K)),
            })

        return pd.DataFrame(out).sort_values("date").reset_index(drop=True)
    ############################################################################################################################################
    @staticmethod
    def _build_cp(df0: pd.DataFrame) -> pd.Series:
        t = df0["type"].astype(str).str.lower().str.strip()
        cp = pd.Series(pd.NA, index=df0.index, dtype="string")
        cp.loc[t.isin(["c"]) | t.str.contains("call", na=False)] = "C"
        cp.loc[t.isin(["p"]) | t.str.contains("put",  na=False)] = "P"
        return cp

    @staticmethod
    def spline_fill_prices(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("strike")
        x = g["strike"].to_numpy(float)
        y = g["option_price"].to_numpy(float)

        x_u, idx = np.unique(x, return_index=True)
        y_u = y[idx]

        if x_u.size < 4:
            return pd.DataFrame()

        step = np.min(np.diff(x_u))
        if not np.isfinite(step) or step <= 0:
            return pd.DataFrame()

        K_grid = np.arange(x_u.min(), x_u.max() + step, step)

        cs = CubicSpline(x_u, y_u, bc_type="natural", extrapolate=False)
        p_grid = cs(K_grid)

        return pd.DataFrame({"strike": K_grid, "price": p_grid}).dropna(subset=["price"])

    def compute_hourly_spline_surfaces(self, df: pd.DataFrame):
        df0 = df.copy()
        df0.index = pd.to_datetime(df0.index)
        df0["maturity"] = pd.to_datetime(df0["maturity"])
        df0["hour"] = df0.index.floor("h")

        df0["cp"] = self._build_cp(df0)

        rowsC = []
        for (hour, maturity), g in tqdm(df0[df0["cp"] == "C"].groupby(["hour", "maturity"])):
            out = self.spline_fill_prices(g)
            if not out.empty:
                out.insert(0, "maturity", maturity)
                out.insert(0, "hour", hour)
                rowsC.append(out)
        df11 = pd.concat(rowsC, ignore_index=True) if rowsC else pd.DataFrame(columns=["hour","maturity","strike","price"])

        rowsP = []
        for (hour, maturity), g in tqdm(df0[df0["cp"] == "P"].groupby(["hour", "maturity"])):
            out = self.spline_fill_prices(g)
            if not out.empty:
                out.insert(0, "maturity", maturity)
                out.insert(0, "hour", hour)
                rowsP.append(out)
        df12 = pd.concat(rowsP, ignore_index=True) if rowsP else pd.DataFrame(columns=["hour","maturity","strike","price"])

        return df11, df12
    

    #######################################################################################################################################################
    @staticmethod
    def _norm_cdf(x):
        from math import erf, sqrt
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    def bs_price(self, is_call, S, K, T, r, q, vol):
        from math import sqrt, log, exp
        if T <= 0 or vol <= 0 or S <= 0 or K <= 0:
            return np.nan
        disc_r = exp(-r * T)
        disc_q = exp(-q * T)
        vsqrtT = vol * sqrt(T)
        d1 = (log(S / K) + (r - q + 0.5 * vol * vol) * T) / vsqrtT
        d2 = d1 - vsqrtT
        if is_call:
            return S * disc_q * self._norm_cdf(d1) - K * disc_r * self._norm_cdf(d2)
        return K * disc_r * self._norm_cdf(-d2) - S * disc_q * self._norm_cdf(-d1)

    def bs_implied_vol_bisect(self, is_call, S, K, T, r, q, price,
                              vol_lo=1e-6, vol_hi=5.0, max_iter=80, tol=1e-8):
        if T <= 0 or price <= 0:
            return np.nan
        lo, hi = vol_lo, vol_hi
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            pm = self.bs_price(is_call, S, K, T, r, q, mid)
            if not np.isfinite(pm):
                return np.nan
            if abs(pm - price) < tol:
                return mid
            if pm < price:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def mfiv_var_from_Q(self, K, Q, r, T_years, F,
                        m_low=0.7, m_high=1.3, n_tail=200):
        K = np.asarray(K, float)
        Q = np.asarray(Q, float)

        ok = np.isfinite(K) & np.isfinite(Q) & (K > 0) & (Q >= 0)
        K = K[ok]; Q = Q[ok]
        if K.size < 5:
            return np.nan

        idx = np.argsort(K)
        K = K[idx]; Q = Q[idx]

        K_lo = F * m_low
        K_hi = F * m_high

        left_tail  = np.logspace(np.log10(K_lo), np.log10(K[0]),  n_tail, endpoint=False) if K_lo < K[0] else []
        right_tail = np.logspace(np.log10(K[-1]), np.log10(K_hi), n_tail)                 if K_hi > K[-1] else []

        K_ext = np.unique(np.concatenate([left_tail, K, right_tail]))
        Q_ext = np.interp(K_ext, K, Q, left=Q[0], right=Q[-1])

        integrand = Q_ext / (K_ext ** 2)
        integral = np.trapezoid(integrand, K_ext)

        var = (2.0 * np.exp(r * T_years) / T_years) * integral
        return var if np.isfinite(var) and var > 0 else np.nan

    def build_MFIV_byH(self, df11, df12, d_btc, H_LIST, q=0.0):
        d_btc = d_btc.copy()
        d_btc.index = pd.to_datetime(d_btc.index).tz_localize(None)

        MFIV_byH = {}

        for H in H_LIST:
            T_years = H / (24.0 * 365.0)
            rows = []

            CP = pd.merge(
                df11, df12,
                on=["hour", "maturity", "strike"],
                how="outer",
                suffixes=("_C", "_P")
            )
            CP["hour"] = pd.to_datetime(CP["hour"]).dt.tz_localize(None)
            CP["maturity"] = pd.to_datetime(CP["maturity"]).dt.tz_localize(None)
            CP["T_hours"] = (CP["maturity"] - CP["hour"]).dt.total_seconds() / 3600.0

            for hour, g in tqdm(CP.groupby("hour"), total=CP["hour"].nunique()):
                if hour not in d_btc.index:
                    continue

                S = float(d_btc.loc[hour, "Close"])
                r = float(d_btc.loc[hour, "rfr"])
                F = float(d_btc.loc[hour, f"F{H}"])

                g = g.loc[(g["T_hours"] - H).abs().groupby(g["strike"]).idxmin()]

                K = g["strike"].to_numpy(float)
                C = g["price_C"].to_numpy(float)
                P = g["price_P"].to_numpy(float)

                Q = np.full_like(K, np.nan)
                Q[K < F] = P[K < F]
                Q[K > F] = C[K > F]

                m = K / F
                if m.min() > 0.85 or m.max() < 1.15:
                    continue

                mfiv_var = self.mfiv_var_from_Q(K, Q, r, T_years, F)
                if not np.isfinite(mfiv_var):
                    continue

                atm_i = np.abs(K - F).argmin()
                px = C[atm_i] if K[atm_i] >= F else P[atm_i]
                is_call = K[atm_i] >= F

                bsIV = self.bs_implied_vol_bisect(is_call, S, K[atm_i], T_years, r, q, px)
                if not np.isfinite(bsIV):
                    continue

                rows.append((hour, bsIV, mfiv_var))

            df_out = pd.DataFrame(rows, columns=["hour", "bsIV", "MFIV"]).set_index("hour").sort_index()
            MFIV_byH[H] = df_out

        return MFIV_byH
