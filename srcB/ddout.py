import json
from urllib.parse import urlencode
import requests
import pandas as pd
from datetime import datetime as dt
from datetime import date, timedelta

def datetime_to_timestamp(datetime_obj):
    return int(dt.timestamp(datetime_obj)*1000)

def timestamp_to_datetime(timestamp):
    return dt.fromtimestamp(timestamp/1000)

def derivative_data(currency: str, kind: str, start_date: date, end_date: date, count: int = 10000) -> pd.DataFrame:
    """Returns derivative trade data for a specified currency and time range."""

    assert isinstance(currency, str), "currency must be a string"
    assert isinstance(start_date, date), "start_date must be a date object"
    assert isinstance(end_date, date), "end_date must be a date object"
    assert start_date <= end_date, "start_date must be before or equal to end_date"

    derivative_list = []
    params = {
        "currency": currency,
        "kind": kind,
        "count": count,
        "include_old": True,
        "start_timestamp": datetime_to_timestamp(dt.combine(start_date, dt.min.time())),
        "end_timestamp": datetime_to_timestamp(dt.combine(end_date, dt.max.time()))
    }

    url = 'https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time'

    with requests.Session() as session:
        while True:
            response = session.get(url, params=params)
            response_data = response.json()
            if len(response_data["result"]["trades"]) == 0:
                break
            derivative_list.extend(response_data["result"]["trades"])
            params["end_timestamp"] = response_data["result"]["trades"][-1]["timestamp"] - 1
            if params["end_timestamp"] <= datetime_to_timestamp(dt.combine(start_date, dt.min.time())):
                break

    df = pd.DataFrame(derivative_list)
    if len(df) == 0:
        return df
    df["date_time"] = pd.to_datetime(df["timestamp"], unit='ms')
    return df