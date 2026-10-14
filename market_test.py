import os, sys, pickle
import pandas as pd
from pathlib import Path

sys.path.insert(
    0,
    "/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/MFIV_calibration"
)

os.chdir("/Users/shah/CODE_BOOK_4/THESIS_WORKING/THESIS_2")

from srcB.market import Market


# =============================================================================================================
# ======================================= DATA ================================================================
# =============================================================================================================

YEAR = 2021

DATA_DIR = Path("../data/rawData")

df = pd.read_parquet(
    DATA_DIR / f"derivative_data_{YEAR}.parquet"
)

d_btc = pd.read_parquet(
    DATA_DIR / f"btc_underlying_minutely_{YEAR}.parquet"
)

d_btc = d_btc.set_index(
    pd.to_datetime(d_btc["date_time"])
)

with open(
    "/Users/shah/CODE_BOOK_4/code_document/REPOSITORIES/data/IS/ready_data/d_btc_by_year_all.pkl",
    "rb"
) as f:
    d_btc_by_year = pickle.load(f)

d_btc["rfr"] = d_btc_by_year[YEAR]["rfr"].reindex(
    d_btc.index,
    method="ffill"
)


# =============================================================================================================
# ======================================= MARKET ==============================================================
# =============================================================================================================

market = Market(
    option_df=df,
    time_df=d_btc,
    starting_cash=100_000,
    im_multiplier=0.15,
    mm_buffer_rate=0.05
)


# =============================================================================================================
# ======================================= CHOOSE TIME =========================================================
# =============================================================================================================

# busiest hour just so there are plenty of contracts to inspect
NOW = market.option["date_time"].value_counts().index[0]

market.update(NOW)

print("\nTIME:", market.now)
print("SPOT:", market.spot)


# =============================================================================================================
# ======================================= SEE AVAILABLE ORDERS ================================================
# =============================================================================================================

available = pd.DataFrame([
    {
        "index": n,
        "symbol": i.Symbol,
        "right": i.Right,
        "direction": i.Direction,
        "strike": i.Strike,
        "expiry": i.Expiry,
        "price": i.Price,
        "amount": i.Amount
    }
    for n, i in enumerate(market.chain.instruments)
])

print(available.head(30).to_string(index=False))


# =============================================================================================================
# ======================================= MANUAL ORDER =========================================================
# =============================================================================================================

CHAIN_INDEX = 0
QTY = 1.0

instrument = market.chain.instruments[CHAIN_INDEX]

print("\nBEFORE")
print("available amount:", instrument.Amount)
print("cash:", market.cash)

if instrument.Direction == "buy":
    order = market.buy(instrument, qty=QTY)
else:
    order = market.sell(instrument, qty=QTY)

print("\nORDER")
print(order)

print("\nAFTER")
print("cash:", market.cash)
print("holdings:", [
    (i.Symbol, i.Direction, i.Amount)
    for i in market.holdings.instruments
])


# =============================================================================================================
# ======================================= FIND NEXT EXIT OPPORTUNITY ==========================================
# =============================================================================================================

holding = market.holdings.instruments[0]

opposite = (
    "sell"
    if holding.Direction == "buy"
    else "buy"
)

future_exit_rows = market.option[
    (market.option["instrument_name"] == holding.Symbol)
    & (market.option["direction"] == opposite)
    & (market.option["date_time"] > market.now)
].sort_values("date_time")

NEXT_NOW = future_exit_rows["date_time"].iloc[0]

market.update(NEXT_NOW)

print("\nEXIT TIME:", NEXT_NOW)

print("available exit volume:", sum(
    i.Amount
    for i in market.chain.instruments
    if (
        i.Symbol == holding.Symbol
        and i.Direction == opposite
    )
))


# =============================================================================================================
# ======================================= MANUAL LIQUIDATION ==================================================
# =============================================================================================================

key = (
    holding.Strike,
    holding.Expiry,
    holding.Right,
    holding.Direction
)

result = market.liquidate(
    key,
    holding
)

print("\nLIQUIDATION RESULT:", result)

print("remaining holdings:", [
    (i.Symbol, i.Direction, i.Amount)
    for i in market.holdings.instruments
])

print("cash:", market.cash)

print("\nTRANSACTIONS")
print(pd.DataFrame(market.transactions).to_string(index=False))