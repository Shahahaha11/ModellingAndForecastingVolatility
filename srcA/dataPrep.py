import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import requests
from arch.utility.exceptions import ConvergenceWarning
from srcA.ddout import derivative_data, datetime_to_timestamp, timestamp_to_datetime

class dataprep:
    def __init__(self, d_btc: pd.DataFrame, df: pd.DataFrame):
        self.d_btc = d_btc.copy()
        self.df = df.copy()

        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message="y is poorly scaled")
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

    def prepare_d_btc(self, rfr):
        d_btc = self.d_btc

        if "date_time" in d_btc.columns:
            d_btc = d_btc.set_index("date_time", drop=False)

        d_btc["rfr"] = (
            rfr.tz_localize("UTC")
               .reindex(d_btc.index.normalize(), method="ffill")
               .to_numpy()
        )

        d_btc = d_btc.resample("1h", label = "right", closed = "right").last()
        # default above is left left.
        d_btc["r"] = np.log(d_btc["close"]).diff()
        d_btc = d_btc.rename(columns={"close": "Close"})

        q = 0.0
        # HOURS = [3, 7, 11, 16, 22, 29, 37, 46, 56, 67, 79, 92, 106, 121, 137, 154, 172, 191] # previous 
        HOURS = [191, 382, 573, 764] # thats 32 days

        for H in HOURS:
            T = H / (24.0 * 365.0)  # years
            d_btc[f"F{H}"] = d_btc["Close"] * np.exp((d_btc["rfr"] - q) * T)

        self.d_btc = d_btc
        return d_btc

    def prepare_df(self, rfr):
        df = self.df

        df["strike"] = df["instrument_name"].str.split("-").str[2].astype(float)
        df["type"] = df["instrument_name"].str.split("-").str[3].astype(str)
        df["option_price"] = df.price * df.index_price

        df["maturity"] = pd.to_datetime(
            df.instrument_name.str.split("-").str[1],
            format="%d%b%y"
        )

        if "date_time" in df.columns:
            df = df.set_index("date_time", drop=False)


        df["rfr"] = rfr.reindex(df.index, method="ffill")

        self.df = df
        return df
    
    @staticmethod
    def get_rfr(d_btc_index, df_index):
        d_btc_index = pd.to_datetime(d_btc_index)
        df_index = pd.to_datetime(df_index)
        d_btc_index = d_btc_index.tz_localize(None) if d_btc_index.tz is None else d_btc_index.tz_convert(None)
        df_index = df_index.tz_localize(None) if df_index.tz is None else df_index.tz_convert(None)

        start = min(d_btc_index.min(), df_index.min()).date() - pd.Timedelta("1D")
        end   = max(d_btc_index.max(), df_index.max()).date()

        start_ts = int(pd.Timestamp(start).timestamp() * 1000)
        end_ts   = int(pd.Timestamp(end).timestamp() * 1000)

        url = "https://www.deribit.com/api/v2/public/get_funding_rate_history"
        records = []
        cur_start = start_ts
        step_ms = 30 * 24 * 3600 * 1000  # request in 30-day chunks, not open-ended

        with requests.Session() as session:
            while cur_start < end_ts:
                chunk_end = min(cur_start + step_ms, end_ts)  # <-- bounded window, not end_ts every time

                params = {
                    "instrument_name": "BTC-PERPETUAL",
                    "start_timestamp": cur_start,
                    "end_timestamp": chunk_end,
                }
                resp = session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                result = resp.json().get("result", [])

                if not result:
                    cur_start = chunk_end + 1
                    continue

                records.extend(result)
                cur_start = max(result[-1]["timestamp"] + 1, chunk_end + 1)

        fund = pd.DataFrame(records)
        fund["date_time"] = pd.to_datetime(fund["timestamp"], unit="ms")
        fund = fund.drop_duplicates(subset="date_time").set_index("date_time").sort_index()

        fund["r_annual"] = fund["interest_1h"] * 24 * 365

        rfr = fund["r_annual"].resample("1D").mean().dropna()
        return rfr    
        
    def run_all(self):
        rfr = self.get_rfr(self.d_btc.index, self.df.index)
        self.prepare_d_btc(rfr)
        self.prepare_df(rfr)
        return self.d_btc, self.df

# ===== Usage =====
# prep = DataPrep2025(d_btc=d_btc, df=df)
# d_btc, df = prep.run_all()
# file_path0, file_path1 = prep.save_ready_frames()
