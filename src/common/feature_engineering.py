# -*- coding: utf-8 -*-
"""Feature engineering module.

This module contains the functions for new data feature addition.
"""
from typing import Literal

import pandas


def __add_sma(data: pandas.DataFrame, col: str, window: int) -> pandas.DataFrame:
    """Adds Simple Moving Average (SMA) columns to the DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame
        col (str): The column to apply SMA
        window (int): The window size for SMA calculation

    Returns:
        pandas.DataFrame: data with new SMA columns
    """
    data[f"{col}_sma"] = data[col].rolling(window=window).mean()
    return data


def __add_wma(data: pandas.DataFrame, col: str, window: int) -> pandas.DataFrame:
    """Adds Weighted Moving Average (WMA) columns to the DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame
        col (str): The column to apply WMA
        window (int): The window size for WMA calculation

    Returns:
        pandas.DataFrame: data with new WMA columns
    """
    weights = range(1, window + 1)
    data[f"{col}_wma"] = (
        data[col].rolling(window=window).apply(lambda prices: sum(weights * prices) / sum(weights), raw=True)
    )
    return data


def __add_ema(data: pandas.DataFrame, col: str, span: int) -> pandas.DataFrame:
    """Adds Exponential Moving Average (EMA) columns to the DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame
        col (str): The column to apply EMA
        span (int): The span size for EMA calculation

    Returns:
        pandas.DataFrame: data with new EMA columns
    """
    data[f"{col}_ema"] = data[col].ewm(span=span, adjust=False).mean()
    return data


def __add_rsi(data: pandas.DataFrame, col: str, window: int) -> pandas.DataFrame:
    """Adds Relative Strength Index (RSI) columns to the DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame
        col (str): The column to apply RSI
        window (int): The window size for RSI calculation

    Returns:
        pandas.DataFrame: data with new RSI columns
    """
    delta = data[col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data[f"{col}_rsi"] = 100 - (100 / (1 + rs))
    return data


def __add_bollinger_bands(data: pandas.DataFrame, col: str, window: int, num_std_dev: int) -> pandas.DataFrame:
    """Adds Bollinger Bands columns to the DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame
        col (str): List of columns to apply SMA
        window (int): The window size for Bollinger Bands calculation
        num_std_dev (int): The number of standard deviations for Bollinger Bands calculation

    Returns:
        pandas.DataFrame: data with new Bollinger Bands columns
    """
    sma = data[col].rolling(window=window).mean()
    std_dev = data[col].rolling(window=window).std()
    data[f"{col}_bb_upper"] = sma + (std_dev * num_std_dev)
    data[f"{col}_bb_lower"] = sma - (std_dev * num_std_dev)
    return data


def __add_lag_features(data: pandas.DataFrame, col: str, lags: list[int]) -> pandas.DataFrame:
    """Adds lag features to the DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame
        columns (list): List of columns to create lag features
        lags (list): List of lag periods to create

    Returns:
        pandas.DataFrame: data with new lag feature columns
    """
    for lag in lags:
        data[f"{col}_lag_{lag}"] = data[col].shift(lag)
    return data


def __add_volatility(data: pandas.DataFrame) -> pandas.DataFrame:
    """Add volatility features to the DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame

    Returns:
        pandas.DataFrame: data with new volatility columns
    """
    if "high" not in data.columns or "low" not in data.columns:
        raise ValueError("Columns 'high' and 'low' not found in DataFrame.")

    data["volatility"] = data["high"] - data["low"]
    return data


def __add_stock_trade_value(data: pandas.DataFrame) -> pandas.DataFrame:
    """Add stock trade value to the DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame

    Returns:
        pandas.DataFrame: data with new stock trade value columns
    """
    if not all(col in data.columns.tolist() for col in ["high", "low", "volume"]):
        raise ValueError("Columns 'high', 'low', and 'volume' not found in DataFrame.")

    data["trade_value"] = (data["high"] + data["low"]) / 2 * data["volume"]
    return data


def add_feature(
    feature: Literal[
        "SMA",
        "WMA",
        "EMA",
        "RSI",
        "Bollinger_Bands",
        "Lag",
        "Volatility",
        "Stock_Trade_Value",
    ],
    data: pandas.DataFrame,
    cols: list[str] | str = None,
    **kwargs,
) -> pandas.DataFrame:
    """Add a feature to the DataFrame.

    Args:
        feature (str): The feature to add
        data (pandas.DataFrame): The input DataFrame
        cols (list | str): The columns to apply the feature
        **kwargs: The keyword arguments for the feature function:
            - window: int
            - span: int
            - num_std_dev: int
            - lags: list[int]

    Returns:
        pandas.DataFrame: data with new feature columns
    """

    function = {
        "SMA": __add_sma,
        "WMA": __add_wma,
        "EMA": __add_ema,
        "RSI": __add_rsi,
        "Bollinger_Bands": __add_bollinger_bands,
        "Lag": __add_lag_features,
        "Volatility": __add_volatility,
        "Stock_Trade_Value": __add_stock_trade_value,
    }

    if isinstance(cols, str):
        cols = [cols]

    fx = function.get(feature, None)
    if fx is None:
        raise ValueError(f"Feature '{feature}' not valid. Valid features are: {list(function.keys())}")

    # Stock Trade Value and Volatility features do not require columns
    if cols is None:
        return fx(data=data)

    # Raise exception if cols have an element not in data_copy.columns
    if not all(col in data.columns.tolist() for col in cols):
        not_cols = [col for col in cols if col not in data.columns.tolist()]
        raise ValueError(f"Columns '{not_cols}' not found in DataFrame.")

    for col in cols:
        data = fx(data=data, col=col, **kwargs)

    return data
