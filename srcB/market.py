# RFR FIXED 11 AUG
 
import pandas as pd
import numpy as np
from scipy.stats import norm
 
 
class Chain:
    def __init__(self):
        self.instruments = []
 
class Instrument:
    def __init__(self, symbol, strike, right, direction, price_per_contract, expiry, option_price_btc):
        self.Symbol = symbol
        self.Strike = strike
        self.Right = right
        self.Direction = direction
        self.Price = price_per_contract
        self.Expiry = expiry
        self.OptionPriceBTC = option_price_btc
 
class Holdings: 
    def __init__(self, cash, now):
        self.time = now
        self.instruments = [] 
        self.cash = cash
        
    def get_cash(self):
        return self.cash
 
    def get_time(self):
        return self.time
        
    def long_instrument(self, instrument):
        self.instruments.append(instrument)
 
    def short_instrument(self, instrument):
        self.instruments.append(instrument)
 
    def close_position(self, key, qty=1):
        # FIX: only remove `qty` matching contracts, not every contract matching the key.
        strike, expiry, right, direction = key
        remaining = []
        removed = 0
        for i in self.instruments:
            if (removed < qty and i.Strike == strike and i.Expiry == expiry
                    and i.Right == right and i.Direction == direction):
                removed += 1
                continue
            remaining.append(i)
        self.instruments = remaining
 
class Market:
    def __init__(self, option_df, time_df, starting_cash=100_000, im_multiplier=0.15, mm_buffer_rate=0.05):
        self.time = time_df.index
        self.option = option_df.copy()
        self.chain = Chain()
        self.option_values = []
        self.time_df = time_df.copy()
        self.spot = None

        # Margin configuration parameters
        self.im_multiplier = im_multiplier    # Standard: 0.10 to 0.15 (10-15%)
        self.mm_buffer_rate = mm_buffer_rate  # Additional market maker buffer standard: 0.05 (5%)
        
        # Live tracking variables
        self.locked_initial_margin = 0.0
        self.locked_buffer_margin = 0.0
                
        self.option["strike"] = self.option["instrument_name"].str.split("-").str[2].astype(float)
        self.option["type"] = self.option["instrument_name"].str.split("-").str[3]
        self.option["maturity"] = pd.to_datetime(
            self.option["instrument_name"].str.split("-").str[1], format="%d%b%y").dt.tz_localize("UTC").dt.floor("h")
        self.option["price_per_contract"] = self.option["price"] * self.option["index_price"]
 
        self.option["date_time"] = pd.to_datetime(self.option["date_time"]).dt.tz_localize("UTC").dt.floor("h")
    
        self.orders = []
        self.transactions = []
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.equity_history = []
 
        self.now = None
        self.holdings = Holdings(starting_cash, self.now)   
        self.current = pd.DataFrame()

    def calculate_margin_requirements(self):
        # Counts only active short positions to lock margin capital
        short_qty = sum(1 for i in self.holdings.instruments if i.Direction == "sell")
        
        # FIXED: Removed contract_size completely; uses your 100x multiplier scale natively
        self.locked_initial_margin = self.spot * self.im_multiplier * short_qty
        self.locked_buffer_margin = self.spot * self.mm_buffer_rate  * short_qty
        

    def get_total_required_margin(self):
        """Returns the total absolute cash balance that must remain in reserves."""
        return self.locked_initial_margin + self.locked_buffer_margin

    def get_free_liquidity(self):
        """Spendable/usable cash balance remaining after subtracting locked margin reserves."""
        return self.cash - self.get_total_required_margin()

    def update(self, now):
        self.now = pd.Timestamp(now).floor("h")
        self.holdings.time = self.now
        self.current = self.option[self.option["date_time"] == self.now].copy()
        self.spot = self.time_df.loc[self.now, "close"]
        
        # FIXED: Reordered processing sequentially so spot exists before margin math triggers
        self.remove_expired()
        self.calculate_margin_requirements()

        self.chain.instruments = [
            Instrument(
                row["instrument_name"],
                row["strike"],
                row["type"],
                row["direction"],
                row["price_per_contract"],
                row["maturity"],
                row["price"]
            )
            for _, row in self.current.iterrows()
        ]
 
    def remove_expired(self):
        expired_options = [i for i in self.holdings.instruments if i.Expiry <= self.now]
 
        for i in expired_options:
            if i.Right == "C":
                payoff = max(self.spot - i.Strike, 0.0)
            else:
                payoff = max(i.Strike - self.spot, 0.0)
 
            delivery_fee = 0.0
            if payoff > 0:
                base_fee = 0.00015 * self.spot 
                fee_cap = 0.125 * payoff
                delivery_fee = min(base_fee, fee_cap)
                    
            if i.Direction == "buy":
                cash_change = (payoff) - delivery_fee
            else:
                cash_change = -(payoff) - delivery_fee
 
            self.transactions.append({
                "time": self.now,
                "action": "expiry_settlement",
                "symbol": i.Symbol,
                "strike": i.Strike,
                "expiry": i.Expiry,
                "right": i.Right,
                "held_direction": i.Direction,
                "qty": 1,
                "price": payoff,
                "fee": delivery_fee,
                "cash_change": cash_change,
                "cash_before": self.cash,
                "cash_after": self.cash + cash_change,
                "spot": self.spot
            })
 
            self.cash += cash_change
 
        self.holdings.instruments = [i for i in self.holdings.instruments if i.Expiry > self.now]
        self.holdings.cash = self.cash
        self.calculate_margin_requirements()
    
    def buy(self, instrument, qty=1):
        fee = min(0.0003, 0.125 * instrument.OptionPriceBTC) * self.spot * qty
        cash_change = -(instrument.Price * qty + fee)

        if self.get_free_liquidity() + cash_change < 0:
            return None
        # if self.cash + cash_change < 0:
        #     return None

        order = {
            "time": self.now,
            "symbol": instrument.Symbol,
            "strike": instrument.Strike,
            "expiry": instrument.Expiry,
            "right": instrument.Right,
            "spot_entry": self.spot,
            "side": "long",
            "qty": qty,
            "entry_price": instrument.Price
        }
        self.orders.append(order)

        self.transactions.append({
            "time": self.now,
            "action": "buy_to_open",
            "symbol": instrument.Symbol,
            "strike": instrument.Strike,
            "expiry": instrument.Expiry,
            "right": instrument.Right,
            "held_direction": "buy",
            "qty": qty,
            "price": instrument.Price,
            "fee": fee,
            "cash_change": cash_change,
            "cash_before": self.cash,
            "cash_after": self.cash + cash_change,
            "spot": self.spot
        })

        self.cash += cash_change
        self.holdings.cash = self.cash

        for _ in range(qty):
            self.holdings.instruments.append(
                Instrument(
                    instrument.Symbol,
                    instrument.Strike,
                    instrument.Right,
                    "buy",
                    instrument.Price,
                    instrument.Expiry,
                    instrument.OptionPriceBTC
                )
            )

        self.calculate_margin_requirements()
        return order
        
    def sell(self, instrument, qty=1):
        fee = min(0.0003, 0.125 * instrument.OptionPriceBTC) * self.spot * qty
        cash_change = instrument.Price * qty - fee

        new_initial_margin = self.spot * self.im_multiplier * qty
        new_buffer_margin = self.spot * self.mm_buffer_rate * qty

        if (self.cash + cash_change - self.locked_initial_margin - self.locked_buffer_margin - new_initial_margin - new_buffer_margin < 0):
            return None
        
        order = {
            "time": self.now,
            "symbol": instrument.Symbol,
            "strike": instrument.Strike,
            "expiry": instrument.Expiry,
            "right": instrument.Right,
            "spot_entry": self.spot,
            "side": "short",
            "qty": qty,
            "entry_price": instrument.Price
        }
        self.orders.append(order)

        self.transactions.append({
            "time": self.now,
            "action": "sell_to_open",
            "symbol": instrument.Symbol,
            "strike": instrument.Strike,
            "expiry": instrument.Expiry,
            "right": instrument.Right,
            "held_direction": "sell",
            "qty": qty,
            "price": instrument.Price,
            "fee": fee,
            "cash_change": cash_change,
            "cash_before": self.cash,
            "cash_after": self.cash + cash_change,
            "spot": self.spot
        })

        self.cash += cash_change
        self.holdings.cash = self.cash

        for _ in range(qty):
            self.holdings.instruments.append(
                Instrument(
                    instrument.Symbol,
                    instrument.Strike,
                    instrument.Right,
                    "sell",
                    instrument.Price,
                    instrument.Expiry,
                    instrument.OptionPriceBTC
                )
            )

        self.calculate_margin_requirements()
        return order
     
    # def liquidate(self, key, instrument, qty=1):
    #     if instrument.Direction == "buy":
    #         action_name = "sell_to_close"
    #         exec_dir = "sell"
    #     else:
    #         action_name = "buy_to_close"
    #         exec_dir = "buy"
 
    #     current_inst = next((i for i in self.chain.instruments if i.Symbol == instrument.Symbol), None)
    #     current_price = current_inst.Price if current_inst else instrument.Price
    #     opt_price_btc = current_inst.OptionPriceBTC if current_inst else instrument.OptionPriceBTC
 
    #     fee = min(0.0003, 0.125 * opt_price_btc) * self.spot * qty
        
    #     if exec_dir == "sell":
    #         cash_change = (current_price * qty) - fee
    #     else:
    #         cash_change = -(current_price * qty + fee)
 
    #     self.transactions.append({
    #         "time": self.now,
    #         "action": action_name,
    #         "symbol": instrument.Symbol,
    #         "strike": instrument.Strike,
    #         "expiry": instrument.Expiry,
    #         "right": instrument.Right,
    #         "held_direction": instrument.Direction,
    #         "qty": qty,
    #         "price": current_price,
    #         "fee": fee,
    #         "cash_change": cash_change,
    #         "cash_before": self.cash,
    #         "cash_after": self.cash + cash_change,
    #         "spot": self.spot
    #     })
 
    #     self.cash += cash_change
    #     self.holdings.close_position(key, qty)
    #     self.holdings.cash = self.cash
    #     self.calculate_margin_requirements()
    def liquidate(self, key, instrument, qty=1):
        if instrument.Direction == "buy":
            action_name = "sell_to_close"
            exec_dir = "sell"
        else:
            action_name = "buy_to_close"
            exec_dir = "buy"

        opposite_direction = "sell" if instrument.Direction == "buy" else "buy"

        current_inst = next(
            (
                i for i in self.chain.instruments
                if (
                    i.Strike == instrument.Strike
                    and i.Expiry == instrument.Expiry
                    and i.Right == instrument.Right
                    and i.Direction == opposite_direction
                )
            ),
            None
        )

        if current_inst is None:
            return None

        current_price = current_inst.Price
        opt_price_btc = current_inst.OptionPriceBTC

        fee = min(
            0.0003,
            0.125 * opt_price_btc
        ) * self.spot * qty

        if exec_dir == "sell":
            cash_change = current_price * qty - fee
        else:
            cash_change = -(current_price * qty + fee)

        self.transactions.append({
            "time": self.now,
            "action": action_name,
            "symbol": instrument.Symbol,
            "strike": instrument.Strike,
            "expiry": instrument.Expiry,
            "right": instrument.Right,
            "held_direction": instrument.Direction,
            "qty": qty,
            "price": current_price,
            "fee": fee,
            "cash_change": cash_change,
            "cash_before": self.cash,
            "cash_after": self.cash + cash_change,
            "spot": self.spot
        })

        self.cash += cash_change
        self.holdings.close_position(key, qty)
        self.holdings.cash = self.cash
        self.calculate_margin_requirements()

        return True
                
    def get_equity_history(self):
        if not self.equity_history:
            return pd.DataFrame(columns=[
                "option_value",
                "cash",
                "equity",
                "required_margin",
                "free_liquidity"
            ])

        return pd.DataFrame(self.equity_history).set_index("time")
    
    def record_equity(self):
        option_value = self.option_value()
        self.option_values.append(option_value)

        self.equity_history.append({
            "time": self.now,
            "option_value": option_value,
            "cash": self.cash,
            "equity": self.cash + option_value,
            "required_margin": self.get_total_required_margin(),
            "free_liquidity": self.get_free_liquidity()
        })
 
    def get_order(self):
        return self.orders
 
    def in_position(self):
        return len(self.holdings.instruments) > 0   
 
    def close_price(self, n=60):
        return self.time_df["close"].loc[:self.now].tail(n)
 
    def variance(self, n=60):
        close = self.close_price(n)
        ret = np.log(close / close.shift(1)).dropna()
        if len(ret) == 0:
            return np.nan
        return ret.var() * (365 * 24)
 
    def std(self, n=60):
        close = self.close_price(n)
        ret = np.log(close / close.shift(1)).dropna()
        if len(ret) == 0:
            return np.nan
        return ret.std()
 
    def option_value(self):
        option_value = 0.0
 
        if self.now is None or not self.holdings.instruments:
            return 0.0
        spot = self.close_price(n=1).iloc[-1]
        var = self.variance(n=60)
        r = float(self.time_df.loc[self.now, "rfr"])
 
        if pd.isna(var) or var <= 0 or pd.isna(spot):
            return 0.0
 
        for instrument in self.holdings.instruments:
            strike = instrument.Strike
            tau = (instrument.Expiry - self.now) / pd.Timedelta(days=365)
 
            if tau <= 0:
                if instrument.Right == "C":
                    price_now = max(spot - strike, 0.0)
                else:
                    price_now = max(strike - spot, 0.0)
            else:
                den = np.sqrt(var) * np.sqrt(tau)
                if den <= 0:
                    price_now = 0.0
                else:
                    d1 = (np.log(spot / strike) + (r + var / 2) * tau) / den
                    d2 = d1 - den
                    Nd1 = norm.cdf(d1)
                    Nd2 = norm.cdf(d2)
                    Nd1_put = norm.cdf(-d1)
                    Nd2_put = norm.cdf(-d2)
 
                    if instrument.Right == "C":
                        price_now = spot * Nd1 - strike * np.exp(-r * tau) * Nd2
                    else:
                        price_now = strike * np.exp(-r * tau) * Nd2_put - spot * Nd1_put
 
            if instrument.Direction == "sell":
                option_value -= price_now
            else:
                option_value += price_now
 
        return option_value
 
    def pricer(self, key):
        strike, expiry, right, direction = key
 
        if self.now is None:
            return 0.0
 
        spot = self.close_price(n=1).iloc[-1]
        var = self.variance(n=60)
        # r = 0.01
        r = float(self.time_df.loc[self.now, "rfr"])
 
        if pd.isna(var) or var <= 0 or pd.isna(spot):
            return 0.0
 
        instrument = next(
            (i for i in self.holdings.instruments
            if i.Strike == strike and i.Expiry == expiry and i.Right == right and
            i.Direction == direction),
            None
        )
 
        if instrument is None:
            return 0.0
 
        tau = (instrument.Expiry - self.now) / pd.Timedelta(days=365)
 
        if tau <= 0:
            if instrument.Right == "C":
                price_now = max(spot - strike, 0.0)
            else:
                price_now = max(strike - spot, 0.0)
        else:
            den = np.sqrt(var) * np.sqrt(tau)
            if den <= 0:
                price_now = 0.0
            else:
                d1 = (np.log(spot / strike) + (r + var / 2) * tau) / den
                d2 = d1 - den
                Nd1 = norm.cdf(d1)
                Nd2 = norm.cdf(d2)
                Nd1_put = norm.cdf(-d1)
                Nd2_put = norm.cdf(-d2)
 
                if instrument.Right == "C":
                    price_now = spot * Nd1 - strike * np.exp(-r * tau) * Nd2
                else:
                    price_now = strike * np.exp(-r * tau) * Nd2_put - spot * Nd1_put
 
        if instrument.Direction == "sell":
            return -price_now
        else:
            return price_now 

            