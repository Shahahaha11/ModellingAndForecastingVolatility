import os
os.chdir(
    "/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/THESIS_PROJECTS/MFIV_calibration"
    )
from srcB.tcn_model_mfiv_oq import (
    _prepare_option_data,
    stage_diagnoses,
    _build_quote_groups,
    _prepare_time_data,
    )
from srcB.tcn_model_mfiv import (
    _entry_contracts,
    _option_price,
    _exit_iv,
    )

# Goal is to find an option with the same keys in the quote groups, check its price and compare it to the price of the actual opposite quote.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#------------------ LOAD OPTION DATA -----------------------
# GLOBAL PARAMS
oneyear = 2021
years = [2019, 2020, 2021]
#------------------ LOAD BOTH DATASETS -----------------------
file_path = "../data/rawData"

df = {}
for y in years :
    df[y] = pd.read_parquet(f"{file_path}/derivative_data_{y}.parquet")

d = {}
for y in years :
    d[y] = pd.read_parquet(f"{file_path}/btc_underlying_minutely_{y}.parquet")


# PREPARE BOTH DATASETS
options = {}
for y in years:
    options[y] = _prepare_option_data(df[y])

time_df = {}
for y in years:
    time_df[y], _ = _prepare_time_data(d[y])

# Now I have three objects in the env, time_df[y], close_col[y], and option[y]


# Lets first add the rfr column, then lets test only 2021

# --------------------------------- Add RFR column --------------------------------
import pickle
with open ("../data/IS/ready_data/d_btc_by_year_all.pkl", "rb") as f :
    d_btc_by_year = pickle.load(f)


for y in years:
    time_df[y]["rfr"] = d_btc_by_year[y]["rfr"].reindex(time_df[y].index, method="ffill")

# -------------------------------- BEGUN GROUPING ---------------------------------
grouped = options[2021].groupby("date_time", sort=False)

# All one liner ways of seeing grouped [guide]
# key, frame = next(iter(grouped)); print(key);print(frame)
# grouped.first()
# grouped.ngroups
# grouped.size()
# grouped.size().plot(kind="hist", bins = 100)
# --------------------------------------------------------------------------------

# --------- TIME_DF CHECKS -------------------------------------------------------
# lets define one strategy right now 
# all we want is the next quote. with the same key
time_df[2021]
time_df[2021].index.dtype
time_df[2021]["close"]

# ------------------ ACTIONS  ---------------------------------------------------
from srcB.strategies import StraddleStrangle
ACTIONS = [
        StraddleStrangle(name="long_straddle_q1", direction="long", structure="straddle", qty=1.0),
]

# --------- Now set times ----------------------------------------------------------
test_selected = {}
# for now in time_df[2021].index[:10].intersection(grouped.groups.keys()):
for now in sorted(set(time_df[2021].index) & set(grouped.groups)):
    snapshot = grouped.get_group(now)
    if snapshot is None : 
        continue
    for action in ACTIONS:
        spot = time_df[2021].loc[now, "close"]
        rfr = time_df[2021].loc[now, "rfr"]
        target_exit_time = now + pd.Timedelta(hours = 24)
        selected = _entry_contracts(
            snapshot, now, spot, rfr,
            target_exit_time, action
        )
    if selected is None : continue
    test_selected[(now, 
                   selected["call"]["instrument_name"],  
                   selected["call"]["direction"])] = {"qty" :selected["qty_available"], 
                                                      "qty_out" :selected["qty_unavailable"], 
                                                      "now": now,
                                                      "spot": spot, 
                                                      "rfr": rfr,
                                                      "bs_price" : None,
                                                      "expiry" : selected["expiry"]}
    test_selected[(now, 
                   selected["put"]["instrument_name"],
                   selected["put"]["direction"])] = {"qty" :selected["qty_available"], 
                                                     "qty_out" :selected["qty_unavailable"],
                                                      "now": now,
                                                      "spot": spot, 
                                                      "rfr": rfr,
                                                      "bs_price" : None,
                                                      "expiry" : selected["expiry"]}


quote_groups =_build_quote_groups(options[2021])

#----------------- CHECKS -----------------
it = iter(quote_groups.items())
next(it)[1].columns
next(it)[1].index[0]
next(iter(test_selected.items()))[1]
#----------------- CHECKS_END -----------------

def opposite_quote(
    quote_groups,
    symbol,
    direction,
    expiry,
    after,
):
    opp_direction = "sell" if direction == "buy" else "buy"
    quotes = quote_groups.get((symbol, opp_direction))
    if quotes is None:
        return None
    quotes[(quotes.index < expiry) &
           (quotes.index >= after)]
    if quotes.empty:
        return None    
    exit_quote = quotes.iloc[0]
    price = exit_quote["price_per_contract"]
    iv = exit_quote["iv"]
    exit_time = exit_quote.name
    return price, iv, exit_time



for ( now, symbol, direction), selected in test_selected.items():
    
    real_exit = opposite_quote(quote_groups, 
                                       symbol, 
                                       direction, 
                                       selected["expiry"],
                                       now + pd.Timedelta(hours = 24),
                                       )
    if real_exit is None :
        continue
    exit_price, exit_iv, exit_time = real_exit
    strike = float(symbol.split("-")[2])
    right = symbol.split("-")[3]
    spot1 = time_df[2021].loc[exit_time, "close"]
    rfr1 = time_df[2021].loc[exit_time, "rfr"]
    try:
        surface_iv = _exit_iv(
            grouped.get_group(exit_time),
            exit_time,
            selected["expiry"], 
            strike, 
            right)
    except KeyError :
        continue
    selected["market_price"] = exit_price
    selected["bs_price"] = _option_price(
        exit_time,
        spot1, 
        strike, 
        rfr1, 
        surface_iv, 
        selected["expiry"], 
        right)
    selected["exit_iv"] = exit_iv
    selected["surface_iv"] = surface_iv
    selected["exit_time"] = exit_time
    selected["spot1"] = spot1


pricer_test = pd.DataFrame.from_dict(test_selected, orient="index")
# -------------------------------- SAVING
os.makedirs("../data/pricer_test",  exist_ok = True)
with open("../data/pricer_test/pricer_test.pkl", "wb") as f:
    pickle.dump(pricer_test, f)
# -------------------------------- SAVING
    
pricer_test
pricer_test["price_error"] = pricer_test["market_price"] - pricer_test["bs_price"]
pricer_test["perr_by_spot"] = pricer_test["price_error"]/ pricer_test["spot1"]

pricer_test[["market_price", "bs_price"]].plot()
pricer_test.plot.scatter("market_price", "bs_price")