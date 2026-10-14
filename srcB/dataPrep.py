import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import requests
from arch.utility.exceptions import ConvergenceWarning
from srcA.ddout import derivative_data, datetime_to_timestamp, timestamp_to_datetime

class dataprep:
    def __init__(self, d_btc: pd.DataFrame, df: pd.DataFrame, horizons=None):
        self.d_btc = d_btc.copy()
        self.df = df.copy()

        # Horizons (in hours) used to build forward-price columns F{H}.
        # Falls back to the historical fixed set if none is passed, so
        # existing notebooks that don't pass `horizons` keep working.
        self.horizons = list(horizons) if horizons is not None else [191, 382, 573, 764]

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

        d_btc = d_btc.resample("1h", label="right", closed="right").last()
        d_btc["r"] = np.log(d_btc["close"]).diff()
        d_btc = d_btc.rename(columns={"close": "Close"})

        q = 0.0

        # for H in self.horizons:
        #     T = H / (24.0 * 365.0)  # years
        #     d_btc[f"F{H}"] = d_btc["Close"] * np.exp((d_btc["rfr"] - q) * T)
        
        F_cols = {}
        for H in self.horizons:
            T = H / (24.0 * 365.0)
            F_cols[f"F{H}"] = d_btc["Close"] * np.exp((d_btc["rfr"] - q) * T)
        d_btc = pd.concat([d_btc, pd.DataFrame(F_cols, index=d_btc.index)], axis=1)
        
        self.d_btc = d_btc
        return d_btc

    def prepare_df(self, rfr):
        df = self.df

        df["strike"] = df["instrument_name"].str.split("-").str[2].astype(float)
        df["type"] = df["instrument_name"].str.split("-").str[3].astype(str)
        df["option_price"] = df.price * df.index_price

        df["maturity"] = (
            pd.to_datetime(
                df.instrument_name.str.split("-").str[1],
                format="%d%b%y"
            ) 
            + pd.Timedelta(hours=8))

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
        end = max(d_btc_index.max(), df_index.max()).date()

        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1MO"
        rfr = pd.read_csv(url)

        rfr["observation_date"] = pd.to_datetime(rfr["observation_date"])
        rfr["DGS1MO"] = pd.to_numeric(rfr["DGS1MO"], errors="coerce")

        rfr = rfr.set_index("observation_date")["DGS1MO"] / 100.0

        # Treasury CMT is quoted on an investment basis;
        # convert annual yield to continuously compounded rate.
        rfr = 2 * np.log1p(rfr / 2)

        rfr = (
            rfr
            .reindex(pd.date_range(start, end, freq="D"))
            .ffill()
            .dropna()
        )

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
