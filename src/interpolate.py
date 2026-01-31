import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import CubicSpline
from math import erf, sqrt, log, exp


class MFIVPipeline:
    """
    Independent, self-contained pipeline that replicates EXACTLY:
      1) df -> df11/df12 (spline-filled call/put price surfaces)
      2) df11/df12 + d_btc -> MFIV_byH (bsIV + MFIV) for each H in H_LIST
    """

    def __init__(self, df: pd.DataFrame, d_btc: pd.DataFrame, q: float = 0.0):
        self.df = df.copy()
        self.d_btc = d_btc.copy()
        self.q = float(q)

        # match your original assumptions
        self.df.index = pd.to_datetime(self.df.index)
        self.d_btc.index = pd.to_datetime(self.d_btc.index).tz_localize(None)

    # =========================
    # Part A: build df11 / df12
    # =========================
    def _prep_df0(self) -> pd.DataFrame:
        df0 = self.df.copy()
        df0.index = pd.to_datetime(df0.index)
        df0["maturity"] = pd.to_datetime(df0["maturity"])
        df0["hour"] = df0.index.floor("h")

        t = df0["type"].astype(str).str.lower().str.strip()
        df0["cp"] = pd.Series(pd.NA, index=df0.index, dtype="string")
        df0.loc[t.isin(["c"]) | t.str.contains("call", na=False), "cp"] = "C"
        df0.loc[t.isin(["p"]) | t.str.contains("put",  na=False), "cp"] = "P"
        return df0

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

    def build_surfaces(self):
        df0 = self._prep_df0()

        rowsC = []
        for (hour, maturity), g in tqdm(df0[df0["cp"] == "C"].groupby(["hour", "maturity"])):
            out = self.spline_fill_prices(g)
            if not out.empty:
                out.insert(0, "maturity", maturity)
                out.insert(0, "hour", hour)
                rowsC.append(out)

        df11 = pd.concat(rowsC, ignore_index=True) if rowsC else pd.DataFrame(
            columns=["hour", "maturity", "strike", "price"]
        )

        rowsP = []
        for (hour, maturity), g in tqdm(df0[df0["cp"] == "P"].groupby(["hour", "maturity"])):
            out = self.spline_fill_prices(g)
            if not out.empty:
                out.insert(0, "maturity", maturity)
                out.insert(0, "hour", hour)
                rowsP.append(out)

        df12 = pd.concat(rowsP, ignore_index=True) if rowsP else pd.DataFrame(
            columns=["hour", "maturity", "strike", "price"]
        )

        return df11, df12

    # =========================
    # Part B: build MFIV_byH
    # =========================
    @staticmethod
    def _norm_cdf(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    @classmethod
    def bs_price(cls, is_call, S, K, T, r, q, vol):
        if T <= 0 or vol <= 0 or S <= 0 or K <= 0:
            return np.nan
        disc_r = exp(-r * T)
        disc_q = exp(-q * T)
        vsqrtT = vol * sqrt(T)
        d1 = (log(S / K) + (r - q + 0.5 * vol * vol) * T) / vsqrtT
        d2 = d1 - vsqrtT
        if is_call:
            return S * disc_q * cls._norm_cdf(d1) - K * disc_r * cls._norm_cdf(d2)
        return K * disc_r * cls._norm_cdf(-d2) - S * disc_q * cls._norm_cdf(-d1)

    @classmethod
    def bs_implied_vol_bisect(cls, is_call, S, K, T, r, q, price,
                              vol_lo=1e-6, vol_hi=5.0, max_iter=80, tol=1e-8):
        if T <= 0 or price <= 0:
            return np.nan
        lo, hi = vol_lo, vol_hi
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            pm = cls.bs_price(is_call, S, K, T, r, q, mid)
            if not np.isfinite(pm):
                return np.nan
            if abs(pm - price) < tol:
                return mid
            if pm < price:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    @staticmethod
    def mfiv_var_from_Q(K, Q, r, T_years, F, m_low=0.7, m_high=1.3, n_tail=200):
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
        right_tail = np.logspace(np.log10(K[-1]), np.log10(K_hi), n_tail) if K_hi > K[-1] else []

        K_ext = np.unique(np.concatenate([left_tail, K, right_tail]))
        Q_ext = np.interp(K_ext, K, Q, left=Q[0], right=Q[-1])

        integrand = Q_ext / (K_ext ** 2)
        integral = np.trapezoid(integrand, K_ext)

        var = (2.0 * np.exp(r * T_years) / T_years) * integral
        return var if np.isfinite(var) and var > 0 else np.nan

    def build_mfiv_byH(self, df11: pd.DataFrame, df12: pd.DataFrame, H_LIST):
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
                if hour not in self.d_btc.index:
                    continue

                S = float(self.d_btc.loc[hour, "Close"])
                r = float(self.d_btc.loc[hour, "rfr"])
                F = float(self.d_btc.loc[hour, f"F{H}"])

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

                bsIV = self.bs_implied_vol_bisect(is_call, S, K[atm_i], T_years, r, self.q, px)
                if not np.isfinite(bsIV):
                    continue

                rows.append((hour, bsIV, mfiv_var))

            MFIV_byH[H] = (
                pd.DataFrame(rows, columns=["hour", "bsIV", "MFIV"])
                .set_index("hour")
                .sort_index()
            )

        return MFIV_byH

    # =========================
    # One-call runner
    # =========================
    def run(self, H_LIST):
        df11, df12 = self.build_surfaces()
        MFIV_byH = self.build_mfiv_byH(df11, df12, H_LIST)
        return df11, df12, MFIV_byH


