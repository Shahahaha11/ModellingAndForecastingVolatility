import numpy as np
import pandas as pd
from tqdm import tqdm
class MFIVFiniteGridDaily:
    def __init__(
        self,
        anchor_time="15:30",
        window_minutes=10,
        target_days=30,
        maturity_band=(21, 45),
        min_strikes=5,
        r=0.0
    ):
        self.anchor_time = anchor_time
        self.window_minutes = window_minutes
        self.target_days = target_days
        self.maturity_band = maturity_band
        self.min_strikes = min_strikes
        self.r = r

    @staticmethod
    def _is_call(t):
        t = str(t).lower()
        return ("call" in t) or (t == "c")

    def compute_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["date_time"] = pd.to_datetime(d["date_time"])
        d["date"] = d["date_time"].dt.normalize()

        out = []


        # for date, g in d.groupby("date"):
        for date, g in tqdm(d.groupby("date"), total=d["date"].nunique()):
            anchor = pd.Timestamp(date.strftime("%Y-%m-%d") + " " + self.anchor_time)
            w0 = anchor - pd.Timedelta(minutes=self.window_minutes)
            w1 = anchor + pd.Timedelta(minutes=self.window_minutes)

            g = g[(g["date_time"] >= w0) & (g["date_time"] <= w1)]
            if g.empty:
                continue

            # VWAP per option
            g["price"] = g["option_price"].astype(float)
            g["strike"] = g["strike"].astype(float)
            g["is_call"] = g["type"].apply(self._is_call)

            vwap = (
                g.groupby(["strike", "maturity", "is_call"])
                 .agg(price=("price", "mean"),
                      spot=("index_price", "last"))
                 .reset_index()
            )

            # maturity in days
            vwap["maturity"] = pd.to_datetime(vwap["maturity"])
            vwap["ttm_days"] = (vwap["maturity"] - date).dt.days

            vwap = vwap[
                (vwap["ttm_days"] >= self.maturity_band[0]) &
                (vwap["ttm_days"] <= self.maturity_band[1])
            ]
            if vwap.empty:
                continue

            # choose expiry with max strike coverage
            exp = (
                vwap.groupby("ttm_days")["strike"]
                .nunique()
                .idxmax()
            )
            vwap = vwap[vwap["ttm_days"] == exp]

            S = vwap["spot"].iloc[0]
            F = S * np.exp(self.r * exp / 365.0)

            puts = vwap[(~vwap["is_call"]) & (vwap["strike"] < F)]
            calls = vwap[(vwap["is_call"]) & (vwap["strike"] > F)]

            if len(puts) < self.min_strikes or len(calls) < self.min_strikes:
                continue

            grid = pd.concat([puts, calls]).sort_values("strike")

            # finite-grid MFIV approximation
            K = grid["strike"].to_numpy()
            Q = grid["price"].to_numpy()

            dK = np.diff(K)
            Kmid = 0.5 * (K[1:] + K[:-1])
            Qmid = 0.5 * (Q[1:] + Q[:-1])

            mfiv = (2.0 / exp) * np.sum(dK * Qmid / (Kmid ** 2))

            out.append({
                "date": date,
                "mfiv": np.sqrt(mfiv),
                "expiry_days": exp,
                "n_strikes": len(grid)
            })

        return pd.DataFrame(out)
