import requests
import pandas as pd
from datetime import datetime as dt
from datetime import date, timezone


def datetime_to_timestamp(datetime_obj):
    if datetime_obj.tzinfo is None:
        datetime_obj = datetime_obj.replace(tzinfo=timezone.utc)
    return int(datetime_obj.timestamp() * 1000)


def timestamp_to_datetime(timestamp):
    return dt.fromtimestamp(timestamp / 1000, tz=timezone.utc)


def futures_data(
    currency: str,
    start_date: date,
    end_date: date,
    count: int = 1000
) -> pd.DataFrame:

    assert isinstance(currency, str), "currency must be a string"
    assert isinstance(start_date, date), "start_date must be a date object"
    assert isinstance(end_date, date), "end_date must be a date object"
    assert start_date <= end_date, "start_date must be before or equal to end_date"

    futures_list = []

    start_timestamp = datetime_to_timestamp(
        dt.combine(start_date, dt.min.time())
    )

    params = {
        "currency": currency,
        "kind": "future",
        "count": count,
        "include_old": True,
        "sorting": "desc",
        "start_timestamp": start_timestamp,
        "end_timestamp": datetime_to_timestamp(
            dt.combine(end_date, dt.max.time())
        )
    }

    url = "https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"

    with requests.Session() as session:
        while True:
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()

            trades = response.json()["result"]["trades"]

            if not trades:
                break

            chunk = pd.DataFrame(trades)

            chunk = chunk.loc[
                chunk["instrument_name"] != f"{currency}-PERPETUAL"
            ].copy()

            if not chunk.empty:
                chunk["date_time"] = pd.to_datetime(
                    chunk["timestamp"], unit="ms"
                )

                chunk["hour"] = (
                    chunk["date_time"].dt.floor("h")
                    + pd.Timedelta(hours=1)
                )

                chunk = (
                    chunk
                    .sort_values("date_time")
                    .groupby(
                        ["hour", "instrument_name"],
                        as_index=False
                    )
                    .tail(1)
                    .drop(columns="hour")
                )

                futures_list.append(chunk)

            params["end_timestamp"] = trades[-1]["timestamp"] - 1

            if params["end_timestamp"] <= start_timestamp:
                break

    if not futures_list:
        return pd.DataFrame()

    df = pd.concat(futures_list, ignore_index=True)

    # An hour/instrument can be split across API pages.
    df["hour"] = (
        df["date_time"].dt.floor("h")
        + pd.Timedelta(hours=1)
    )

    df = (
        df
        .sort_values("date_time")
        .groupby(
            ["hour", "instrument_name"],
            as_index=False
        )
        .tail(1)
        .drop(columns="hour")
        .sort_values("date_time")
        .reset_index(drop=True)
    )

    return df