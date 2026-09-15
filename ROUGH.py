# SEARCH WORDS :
# FINAL_READY_OUTPUTS_BY_YEAR
# YEARLY_TCN_ESTIMATION
# YEARLY_MFIV_GARCH_ESTIMATION
# TESTING_MAIN_WITH_CACHE
# TCN_SCORE_CALIBRATION_DIAGNOSTIC
## TCN_SCORE_THRESHOLD_CALIBRATION
### TCN_SCORE_THRESHOLD_TRADER_CALIBRATION
import pandas as pd
import numpy as np
from math import sqrt, pi, exp
from arch import arch_model
import yfinance as yf 
from scipy.optimize import minimize
import warnings
from arch.__future__ import reindexing
from arch.utility.exceptions import ConvergenceWarning
# from srcA.msGarch import msGARCH
# from srcA.metrics import metrics
# from srcA.msGarch import msGARCHIV
from tqdm import tqdm

from xgboost import XGBRegressor
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="y is poorly scaled")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


import os, sys
sys.path.insert(0, "/Users/shah/CODE_BOOK_4/THESIS_WORKING/THESIS_2")
sys.path.insert(0, "/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/MFIV_calibration")
os.chdir("/Users/shah/CODE_BOOK_4/THESIS_WORKING/THESIS_2")
sys.modules.pop("srcB", None)

### NOTE :
import pandas as pd

# READ D_BTC AND DF
from pathlib import Path
DATA_DIR = Path("../data/rawData")
YEARS = [2019, 2020, 2021]
d_btc_raw_by_year = {}
df_raw_by_year = {}
for y in YEARS:
    d_btc_raw_by_year[y] = pd.read_parquet(DATA_DIR / f"btc_underlying_minutely_{y}.parquet")
    df_raw_by_year[y]    = pd.read_parquet(DATA_DIR / f"derivative_data_{y}.parquet")

# SET INDEX TO DATETIME
for y in YEARS:
    d_btc_raw_by_year[y] = d_btc_raw_by_year[y].set_index(pd.to_datetime(d_btc_raw_by_year[y]["date_time"]))
    df_raw_by_year[y] = df_raw_by_year[y].set_index(pd.to_datetime(df_raw_by_year[y]["date_time"]))


# CHECKING LOWEST ACCEPTABLE QUANTITY
bad_ratios = {}
for y in YEARS:
    df_raw_by_year[y]
    is_acceptable = df_raw_by_year[y]["amount"] >= 0.1
    bad_ratios[y] =  (~is_acceptable).sum()/df_raw_by_year[y]["amount"].count()
bad_ratios


#=============== PROJECT UNRELATED PYTHON FUNCTION CHECKS ===============
nums = iter([10,20, 30])
print(list(nums))

# lets not destroy the original iterator.
import itertools
nums = iter([10, 20, 30])
nums, inspector = itertools.tee(nums)
print(list(inspector))