# -*- coding: utf-8 -*-
"""Stats module.

This module contains the statistical functions to be used in the project.
"""
import logging

import numpy
import pandas
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller

from common import get_logger


def remove_outliers_iqr(
    dataframe: pandas.DataFrame, columns: str | list[str] | None = None, whisker_width: float = 1.5
) -> pandas.DataFrame:
    """Removes outliers from a dataframe by column, including optional whiskers, removing rows
     for which the column value is less than Q1-1.5IQR or greater than Q3+1.5IQR.

    Args:
        dataframe (`:obj:pandas.DataFrame`): A pandas dataframe to subset
        columns (list[str] | str): Name of the column to calculate the subset from.
        whisker_width (float): Optional, loosen the IQR filter by a factor of `whisker_width` * IQR.

    Returns:
        (`:obj:pd.DataFrame`): Filtered dataframe
    """
    if not columns:
        columns = dataframe.select_dtypes(include="number").columns.tolist()

    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        # Calculate Q1, Q2 and IQR
        q1 = dataframe[col].quantile(0.25)
        q3 = dataframe[col].quantile(0.75)
        iqr = q3 - q1

        # Apply filter with respect to IQR, including optional whiskers
        dataframe[col] = dataframe[col].loc[
            (dataframe[col] >= q1 - whisker_width * iqr) & (dataframe[col] <= q3 + whisker_width * iqr)
        ]

    return dataframe


def __check_stationarity(data: pandas.Series) -> bool:
    """Check stationarity of the time series.

    Args:
        data (pandas.Series): data to check for stationarity

    Returns:
        bool: True if stationary, False for non-stationary
    """
    _, pvalue, *_ = adfuller(x=data)
    # p-value <= 0.05 indicates stationarity
    return pvalue <= 0.05


def check_stationarity(data: pandas.DataFrame, columns: list[str], logger: logging.Logger = None) -> list[str]:
    """Check stationarity of the time series.

    Args:
        data (pandas.DataFrame): data to check for stationarity
        columns (list[str]): columns to check for stationarity

    Returns:
        list[str]: list of non-stationary variables
    """
    if not logger:
        logger = get_logger()

    logger.info("Based upon the significance level of 0.05 and the p-value of the ADF test:\n")
    non_stationary_vars = []
    for var in columns:
        _, pvalue, *_ = adfuller(x=data[var])

        rejected = not __check_stationarity(data[var])
        if rejected:
            non_stationary_vars.append(var)

        logger.info(
            f"- The null hypothesis for {var:^10} can{' not' if rejected else ' '*4} be rejected. Hence, the series is "
            f"{'non-' if rejected else ' '*4}stationary."
        )

    return non_stationary_vars


def calculate_vif(data: pandas.DataFrame) -> pandas.Series:
    """Calculate the Variance Inflation Factor (VIF) for the columns in the data.

    Args:
        data (pandas.DataFrame): data

    Returns:
        pandas.Series: VIF values
    """
    data.dropna(inplace=True)

    num_data = data.select_dtypes(include="number")

    vif = [
        variance_inflation_factor(exog=num_data.values.astype(dtype=float), exog_idx=i)
        for i in range(num_data.shape[1])
    ]

    # replace float(inf)
    vif = numpy.where(numpy.isinf(vif), 1_000_000, vif)

    return pandas.Series(
        data=vif,
        index=num_data.columns,
    )
