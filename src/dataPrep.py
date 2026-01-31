import pandas as pd
import numpy as np
import yfinance as yf
import warnings
from arch.utility.exceptions import ConvergenceWarning

class dataprep:
    def __init__(self, d_btc: pd.DataFrame, df: pd.DataFrame):
        # keep original names
        self.d_btc = d_btc.copy()
        self.df = df.copy()

        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message="y is poorly scaled")
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

    def prepare_d_btc(self):
        d_btc = self.d_btc

        if "date_time" in d_btc.columns:
            d_btc = d_btc.set_index("date_time", drop=False)

        rfr = yf.download(
            "^IRX",
            start=d_btc.index.min().date() - pd.Timedelta("1D"),
            end=d_btc.index.max().date(),
            progress=False
        )["Close"].dropna() / 100

        d_btc["rfr"] = (
            rfr.tz_localize("UTC")
               .reindex(d_btc.index.normalize(), method="ffill")
               .to_numpy()
        )

        d_btc = d_btc.resample("1h").last()
        d_btc["r"] = np.log(d_btc["close"]).diff()
        d_btc = d_btc.rename(columns={"close": "Close"})

        self.d_btc = d_btc
        return d_btc

    def prepare_df(self):
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

        rfr = yf.download(
            "^IRX",
            start=df.index.min().normalize() - pd.Timedelta("1D"),
            end=df.index.max().normalize(),
            progress=False
        )["Close"] / 100

        df["rfr"] = rfr.reindex(df.index, method="ffill")

        self.df = df
        return df

    def run_all(self):
        self.prepare_d_btc()
        self.prepare_df()
        return self.d_btc, self.df

    def save_ready_frames(
        self,
        file_path0: str = "../data/d_2025.parquet",
        file_path1: str = "../data/df_2025.parquet"
    ):
        self.d_btc.to_parquet(file_path0)
        self.df.to_parquet(file_path1)
        return file_path0, file_path1


# ===== Usage =====
# prep = DataPrep2025(d_btc=d_btc, df=df)
# d_btc, df = prep.run_all()
# file_path0, file_path1 = prep.save_ready_frames()
