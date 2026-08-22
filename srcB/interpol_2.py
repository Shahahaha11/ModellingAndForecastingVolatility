import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm
from math import erf, sqrt, log, exp
class MFIVPipeline:
    """
    One class that replicates BOTH cells exactly:
      1) builds df11/df12 via cubic-spline strike filling per (hour, maturity, cp)
      2) builds MFIV_byH via nearest-maturity-to-H per strike, MFIV integral, and ATM BSIV
    """

    def __init__(self, df: pd.DataFrame, d_btc: pd.DataFrame, d_fut: pd.DataFrame, q: float = 0.0):
        self.df = df
        self.d_btc = d_btc.copy()
        self.d_fut = d_fut.copy()
        self.q = q
        self.d_btc.index = pd.to_datetime(self.d_btc.index).tz_localize(None)
        
        self.d_fut["hour"] = pd.to_datetime(self.d_fut["hour"]).dt.tz_localize(None)
        self.d_fut["maturity"] = pd.to_datetime(self.d_fut["maturity"]).dt.tz_localize(None)

        self.fut_by_hour = {}

        for hour, g in self.d_fut.groupby("hour"):
            T_hours = (
                (g["maturity"] - hour).dt.total_seconds() / 3600.0
            ).to_numpy(float)

            F = g["F"].to_numpy(float)

            ok = (
                (T_hours > 0)
                & np.isfinite(F)
                & (F > 0)
            )

            T_hours = T_hours[ok]
            F = F[ok]

            idx = np.argsort(T_hours)

            self.fut_by_hour[hour] = (
                T_hours[idx],
                F[idx]
            )

        self.df11 = None
        self.df12 = None
        self.mfiv_maturity = None
        self.MFIV_byH = None
        
    # =========================
    # Cell 1 helpers
    # =========================
    @staticmethod
    def spline_fill_prices(g: pd.DataFrame, K_grid: np.ndarray) -> pd.DataFrame:
        g = g.sort_values("strike")
        x = g["strike"].to_numpy(float)
        y = g["option_price"].to_numpy(float)

        # unique strikes
        x_u, idx = np.unique(x, return_index=True)
        y_u = y[idx]

        # need enough points
        if x_u.size < 4:
            return pd.DataFrame()

        cs = PchipInterpolator(x_u, y_u, extrapolate=False)
        p_grid = cs(K_grid)

        return pd.DataFrame({"strike": K_grid, "price": p_grid}).dropna(subset=["price"])
    
    def build_surfaces(self): # uses spline fill prices
        # assumes df has: type, strike, maturity, option_price, and datetime index
        df0 = self.df.copy()
        df0.index = pd.to_datetime(df0.index)
        df0["maturity"] = pd.to_datetime(df0["maturity"])
        df0["hour"] = df0.index.floor("h") + pd.Timedelta(hours=1)

        t = df0["type"].astype(str).str.lower().str.strip()
        df0["cp"] = pd.Series(pd.NA, index=df0.index, dtype="string")
        df0.loc[t.isin(["c"]) | t.str.contains("call", na=False), "cp"] = "C"
        df0.loc[t.isin(["p"]) | t.str.contains("put",  na=False), "cp"] = "P"
        
        # ----- df11/df12: call+put price surfaces (filled across a SHARED strike grid) -----
        rowsC, rowsP = [], []
        for (hour, maturity), g_all in tqdm(df0.groupby(["hour", "maturity"])):
            x_u_all = np.unique(g_all["strike"].to_numpy(float))
            if x_u_all.size < 4:
                continue
            step = np.min(np.diff(x_u_all))
            if not np.isfinite(step) or step <= 0:
                continue
            K_grid = np.arange(x_u_all.min(), x_u_all.max() + step, step)

            outC = self.spline_fill_prices(g_all[g_all["cp"] == "C"], K_grid)
            if not outC.empty:
                outC.insert(0, "maturity", maturity)
                outC.insert(0, "hour", hour)
                rowsC.append(outC)

            outP = self.spline_fill_prices(g_all[g_all["cp"] == "P"], K_grid)
            if not outP.empty:
                outP.insert(0, "maturity", maturity)
                outP.insert(0, "hour", hour)
                rowsP.append(outP)

        df11 = pd.concat(rowsC, ignore_index=True) if rowsC else pd.DataFrame(
            columns=["hour", "maturity", "strike", "price"]
        )
        df12 = pd.concat(rowsP, ignore_index=True) if rowsP else pd.DataFrame(
            columns=["hour", "maturity", "strike", "price"]
        )
        self.df11, self.df12 = df11, df12
        return df11, df12

    # =========================
    # Cell 2 helpers (BS + MFIV)
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

        # tail extension in STRIKE, flat PRICE
        K_lo = F * m_low
        K_hi = F * m_high

        left_tail = np.logspace(np.log10(K_lo), np.log10(K[0]), n_tail, endpoint=False) if K_lo < K[0] else []
        right_tail = np.logspace(np.log10(K[-1]), np.log10(K_hi), n_tail) if K_hi > K[-1] else []

        K_ext = np.unique(np.concatenate([left_tail, K, right_tail]))
        Q_ext = np.interp(K_ext, K, Q, left=0.0, right=0.0)

        integrand = Q_ext / (K_ext ** 2)
        integral = np.trapezoid(integrand, K_ext)

        var = (2.0 * np.exp(r * T_years) / T_years) * integral
        return var if np.isfinite(var) and var > 0 else np.nan

    def _forward_for_maturity(self, hour, maturity, S):

        curve = self.fut_by_hour.get(hour)

        if curve is None:
            return np.nan

        T_hours, F = curve

        if len(T_hours) == 0:
            return np.nan

        T_target = (maturity - hour).total_seconds() / 3600.0

        if T_target <= 0 or T_target > T_hours[-1]:
            return np.nan

        T = np.r_[0.0, T_hours]
        logF = np.r_[np.log(S), np.log(F)]

        return float(
            np.exp(np.interp(T_target, T, logF))
        )
    def _mfiv_for_maturity(self, g_mat, r, T_years, F):
        if g_mat.empty or not np.isfinite(F):
            return np.nan
        K = g_mat["strike"].to_numpy(float)
        C = g_mat["price_C"].to_numpy(float)
        P = g_mat["price_P"].to_numpy(float)
        Q = np.full_like(K, np.nan)
        Q[K < F] = P[K < F]
        Q[K > F] = C[K > F]
        m = K / F
        if m.min() > 0.85 or m.max() < 1.15:
            return np.nan
        return self.mfiv_var_from_Q(K, Q, r, T_years, F)

    def precompute_mfiv_maturities(self):
        if self.df11 is None or self.df12 is None:
            raise ValueError("Run build_surfaces() first (needs df11/df12).")

        CP = pd.merge(
            self.df11, self.df12,
            on=["hour", "maturity", "strike"],
            how="outer",
            suffixes=("_C", "_P")
        )

        CP["hour"] = pd.to_datetime(CP["hour"]).dt.tz_localize(None)
        CP["maturity"] = pd.to_datetime(CP["maturity"]).dt.tz_localize(None)

        CP["T_hours"] = (
            (CP["maturity"] - CP["hour"]).dt.total_seconds() / 3600.0
        )

        rows = []

        for hour, g in tqdm(
            CP.groupby("hour"),
            total=CP["hour"].nunique()
        ):
            if hour not in self.d_btc.index:
                continue

            S = float(self.d_btc.loc[hour, "Close"])
            r = float(self.d_btc.loc[hour, "rfr"])

            g = g.loc[g["T_hours"] > 0]

            if g.empty:
                continue

            for maturity, g_mat in g.groupby("maturity"):

                T_hours = float(g_mat["T_hours"].iloc[0])
                T_years = T_hours / 8760.0

                F = self._forward_for_maturity(
                    hour, maturity, S
                )

                mfiv_var = self._mfiv_for_maturity(
                    g_mat, r, T_years, F
                )

                bsIV = np.nan

                if np.isfinite(F):
                    K = g_mat["strike"].to_numpy(float)
                    C = g_mat["price_C"].to_numpy(float)
                    P = g_mat["price_P"].to_numpy(float)

                    atm_i = np.abs(K - F).argmin()

                    px = (
                        C[atm_i]
                        if K[atm_i] >= F
                        else P[atm_i]
                    )

                    is_call = K[atm_i] >= F

                    lower = max(
                        0.0,
                        (S - K[atm_i])
                        if is_call
                        else (K[atm_i] - S)
                    )

                    upper = S if is_call else K[atm_i]

                    if lower <= px <= upper:
                        bsIV = self.bs_implied_vol_bisect(
                            is_call,
                            S,
                            K[atm_i],
                            T_years,
                            r,
                            self.q,
                            px
                        )

                total_var = (
                    mfiv_var * T_years
                    if np.isfinite(mfiv_var)
                    else np.nan
                )

                rows.append(
                    (
                        hour,
                        maturity,
                        T_hours,
                        F,
                        bsIV,
                        mfiv_var,
                        total_var
                    )
                )

        self.mfiv_maturity = (
            pd.DataFrame(
                rows,
                columns=[
                    "hour",
                    "maturity",
                    "T_hours",
                    "F",
                    "bsIV",
                    "MFIV",
                    "total_var"
                ]
            )
            .sort_values(["hour", "T_hours"])
            .reset_index(drop=True)
        )

        return self.mfiv_maturity
        
    def compute_mfiv_byH(self, H_LIST):

        if self.mfiv_maturity is None:
            raise ValueError("Set mfiv_maturity or run precompute_mfiv_maturities() first.")

        rows_by_H = {H: [] for H in H_LIST}

        for hour, g in tqdm(
            self.mfiv_maturity.groupby("hour"),
            total=self.mfiv_maturity["hour"].nunique()
        ):

            g = g.sort_values("T_hours")

            T = g["T_hours"].to_numpy(float)
            mfiv = g["MFIV"].to_numpy(float)
            total_var = g["total_var"].to_numpy(float)
            bsiv = g["bsIV"].to_numpy(float)

            if len(T) == 0:
                continue

            for H in H_LIST:

                nearest_i = np.abs(T - H).argmin()

                mfiv_var = np.nan

                j = np.searchsorted(
                    T,
                    H,
                    side="right"
                )

                if j > 0 and j < len(T):

                    i1 = j - 1
                    i2 = j

                    if (
                        np.isfinite(mfiv[i1])
                        and np.isfinite(mfiv[i2])
                    ):
                        T1 = T[i1] / 8760.0
                        T2 = T[i2] / 8760.0
                        T_target = H / 8760.0

                        total_H = (
                            total_var[i1]
                            + (total_var[i2] - total_var[i1])
                            * (T_target - T1)
                            / (T2 - T1)
                        )

                        mfiv_var = total_H / T_target

                # same nearest-maturity fallback
                if not np.isfinite(mfiv_var):
                    mfiv_var = mfiv[nearest_i]

                if not np.isfinite(mfiv_var):
                    continue

                rows_by_H[H].append(
                    (
                        hour,
                        bsiv[nearest_i],
                        mfiv_var
                    )
                )

        MFIV_byH = {}

        for H in H_LIST:
            MFIV_byH[H] = (
                pd.DataFrame(
                    rows_by_H[H],
                    columns=["hour", "bsIV", "MFIV"]
                )
                .set_index("hour")
                .sort_index()
            )

        self.MFIV_byH = MFIV_byH
        return MFIV_byH
    
    def run(self, H_LIST):
        df11, df12 = self.build_surfaces()
        self.precompute_mfiv_maturities()
        MFIV_byH = self.compute_mfiv_byH(H_LIST)
        return df11, df12, MFIV_byH
