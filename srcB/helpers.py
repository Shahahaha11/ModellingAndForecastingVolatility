import math
import pandas as pd

def norm_pdf(x):
    return math.exp(-x**2 / 2) / math.sqrt(2 * math.pi)

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def get_d1_d2(now, spot, strike, rfr, sigma, expiry):
    sigma = sigma / 100
    tau = (expiry - now) / pd.Timedelta(days=365)
    d1_num = math.log(spot / strike) + (rfr + 0.5 * sigma * sigma) * tau
    d1_den = sigma * math.sqrt(tau)
    d1 = d1_num / d1_den
    d2 = d1 - d1_den
    return d1, d2

def get_delta(d1, right, direction):
    delta = norm_cdf(d1)
    if right == "P":
        delta = delta - 1
    if direction == "sell":
        delta *= -1
    return delta

def get_gamma(d1, right, now, spot, sigma, expiry, direction):
    sigma = sigma / 100
    tau = (expiry - now) / pd.Timedelta(days=365)
    gamma = norm_pdf(d1) / (spot * sigma * math.sqrt(tau))
    if direction == "sell":
        gamma *= -1
    return gamma