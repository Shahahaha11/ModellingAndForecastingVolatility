
import numpy as np
import pandas as pd
from math import log, sqrt, exp, erf

class BSMImpliedVol:
    def __init__(self, day_count: float = 365.0, q: float = 0.0):
        self.day_count = float(day_count)
        self.q = float(q)

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    def _bs_price(self, is_call: bool, S: float, K: float, T: float, r: float, vol: float) -> float:
        if T <= 0 or vol <= 0 or S <= 0 or K <= 0:
            return np.nan
        disc_r = exp(-r * T)
        disc_q = exp(-self.q * T)
        vsqrtT = vol * sqrt(T)
        d1 = (log(S / K) + (r - self.q + 0.5 * vol * vol) * T) / vsqrtT
        d2 = d1 - vsqrtT
        N1 = self._norm_cdf(d1)
        N2 = self._norm_cdf(d2)
        if is_call:
            return S * disc_q * N1 - K * disc_r * N2
        else:
            # put via put formula
            Nm1 = self._norm_cdf(-d1)
            Nm2 = self._norm_cdf(-d2)
            return K * disc_r * Nm2 - S * disc_q * Nm1

    def implied_vol(self, opt_type, dt, maturity, K, price, S, r: float,
                    vol_lo: float = 1e-6, vol_hi: float = 5.0, max_iter: int = 80, tol: float = 1e-8):
        # type -> is_call
        t = str(opt_type).lower().strip()
        is_call = (t == "c") or ("call" in t)

        # T in years
        dt = pd.to_datetime(dt, utc=True, errors="coerce")
        maturity = pd.to_datetime(maturity, utc=True, errors="coerce")
        if pd.isna(dt) or pd.isna(maturity):
            return np.nan

        T = (maturity - dt).total_seconds() / (24.0 * 3600.0) / self.day_count
        if T <= 0:
            return np.nan

        S = float(S); K = float(K); price = float(price); r = float(r)
        if not (S > 0 and K > 0 and price > 0):
            return np.nan

        # No-arb lower bound (rough) to reject impossible quotes
        disc_r = exp(-r * T)
        disc_q = exp(-self.q * T)
        intrinsic = max(0.0, S * disc_q - K * disc_r) if is_call else max(0.0, K * disc_r - S * disc_q)
        if price < intrinsic - 1e-10:
            return np.nan

        # monotone in vol -> bisection
        lo, hi = vol_lo, vol_hi
        plo = self._bs_price(is_call, S, K, T, r, lo)
        phi = self._bs_price(is_call, S, K, T, r, hi)
        if np.isnan(plo) or np.isnan(phi):
            return np.nan

        # expand hi if needed
        tries = 0
        while phi < price and tries < 10:
            hi *= 2.0
            phi = self._bs_price(is_call, S, K, T, r, hi)
            tries += 1
            if hi > 20.0:
                break
        if phi < price:
            return np.nan

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            pm = self._bs_price(is_call, S, K, T, r, mid)
            if np.isnan(pm):
                return np.nan
            if abs(pm - price) <= tol:
                return mid
            if pm < price:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

def add_bs_iv(df: pd.DataFrame, r: float, day_count: float = 365.0, q: float = 0.0) -> pd.DataFrame:
    x = df.copy()
    iv = BSMImpliedVol(day_count=day_count, q=q)

    # compute per row (kept simple; optimize later if needed)
    x["bsIV"] = [
        iv.implied_vol(t, dt, mat, K, p, S, r)
        for t, dt, K, p, mat, S in zip(
            x["type"], x["date_time"], x["strike"], x["option_price"], x["maturity"], x["index_price"]
        )
    ]
    return x

