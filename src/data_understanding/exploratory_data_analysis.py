# -*- coding: utf-8 -*-
"""Exploratory Data Analysis.

This module contains the exploratory data analysis utilities to be used in the project.
"""

import pandas
import seaborn
from matplotlib import pyplot
from statsmodels import api as sm

from common import get_dataset, get_logger, msg_task, save_plot, check_stationarity

logger = get_logger(log_filename="exploratory_data_analysis.log")


def data_analysis(data: pandas.DataFrame, figsize: tuple = (16, 9)) -> None:
    """Data analysis.

    Args:
        data (pandas.DataFrame): data
        figsize (set, optional): figure size. Defaults to (14, 8).
    """
    # Summary
    logger.info(f"Variables summary:\n\n{data.describe().T}\n\n")

    # Missing values
    logger.info(f"Missing values:\n\n{data.isnull().sum()}\n\n")

    # Check for outliers using IQR
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
    logger.info(f"Outliers:\n\n{outliers}\n\n")

    # Plotting
    logger.info("Data Visualization\n\n")

    # Plot close prices
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.plot(
        data.index,
        data["close"],
        label="Close Price",
    )
    ax.set_title("Close Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    save_plot(filename="eda/close_price_over_time.png")
    pyplot.show()

    # Plot all prices together
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.plot(
        data.index,
        data["volume"],
        label="Volume",
        color="orange",
    )
    ax.set_title("Volume Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    save_plot(filename="eda/volume_over_time.png")
    pyplot.show()

    # Plot all prices in 2023
    var_color = {
        "adj_close": "black",
        "open": "green",
        "high": "purple",
        "low": "red",
        "close": "blue",
    }
    data_23 = data["2023-01-01":"2024-01-01"]
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=figsize)
    for var, color in var_color.items():
        ax.plot(data_23.index, data_23[var], label=var.title(), color=color)

    ax.set_title("Price Over 2023")
    ax.set_xlabel("Date")
    pyplot.legend()
    save_plot(filename="eda/price_over_2023.png")
    pyplot.show()

    # Correlation heatmap
    seaborn.heatmap(data.corr(), annot=True, cmap="coolwarm")
    pyplot.title("Correlation Matrix")
    save_plot(filename="eda/correlation_matrix.png")
    pyplot.show()

    # Boxplot for price distribution
    seaborn.boxplot(data=data[["open", "high", "low", "close"]])
    pyplot.title("Price Distribution")
    pyplot.tight_layout()
    save_plot(filename="eda/price_distribution.png")
    pyplot.show()

    # Boxplot for volume distribution
    seaborn.boxplot(data=data["volume"])
    pyplot.title("Volume Distribution")
    pyplot.tight_layout()
    save_plot(filename="eda/volume_distribution.png")
    pyplot.show()


def time_series_analysis(data: pandas.DataFrame, figsize: tuple = (16, 9)) -> None:
    """Time series analysis.

    Args:
        data (pandas.DataFrame): data
        figsize (set, optional): figure size. Defaults to (14, 8).
    """

    # Plot monthly close prices to check for long-term trends
    data_monthly = data.resample("ME").mean()
    rolling_avg_12_months = data_monthly.rolling(window=12).mean()
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=figsize)
    period = slice("2001", "2024")
    data_monthly[period]["close"].plot(
        ax=ax,
        label="Montly Close Price",
        marker=".",
    )
    rolling_avg_12_months[period]["close"].plot(
        label="Long-term Trend (12 Months Rolling Avg)",
        color="red",
    )
    ax.set_title("Monthly Close Price Over 2023")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    pyplot.legend()
    save_plot(filename="eda/monthly_close_price.png")
    pyplot.show()

    logger.info("The plot shows a long-term trend in the close price, and a possible yearly seasonality.\n\n")

    # Seasonal decomposition
    data_monthly.diff(12)[period]["close"].plot(marker=".", figsize=(8, 3), label="Diffenciated Yearly Close Price")
    save_plot(filename="eda/yearly_differentiated_close_price.png")
    pyplot.show()

    # Stationarity test
    logger.info("Stationarity test\n\n")

    non_stationary_vars = check_stationarity(data=data, columns=data.columns.tolist(), logger=logger)

    logger.info(
        f"\n\nAs variables {', '.join(non_stationary_vars)} are non-stationary, we can try to make them stationary by "
        f"detrending them and checking if they become stationary. This can be done by differencing."
    )

    # Differencing
    data_stationarized = data[non_stationary_vars].diff(periods=1).dropna()
    non_stationary_vars_diff = check_stationarity(
        data=data_stationarized, columns=data_stationarized.columns.tolist(), logger=logger
    )

    if len(non_stationary_vars_diff) == 0:
        logger.info(f"\n\nAfter differencing, variables {', '.join(non_stationary_vars)} became stationary.")
    else:
        logger.info(
            f"\n\nAfter differencing, variables {', '.join(non_stationary_vars_diff)} are still non-stationary."
        )

    # Plot close prices differentiated
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.plot(
        data_stationarized.index,
        data_stationarized["close"],
        label="Diffenciated Close Price",
    )
    ax.set_title("Diffenciated Close Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    save_plot(filename="eda/diffenciated_close_price_over_time.png")
    pyplot.show()

    # Plot close prices differentiated 2023
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.plot(
        data_stationarized["2023-01-01":"2024-01-01"].index,
        data_stationarized["2023-01-01":"2024-01-01"]["close"],
        label="Diffenciated Close Price",
    )
    ax.set_title("Diffenciated Close Price Over 2023")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    save_plot(filename="eda/diffenciated_close_price_over_2023.png")
    pyplot.show()

    logger.info("The plot shows a stationary time series. So, now we can check seasonality.\n\n")

    # Autocorrelation plot
    sm.graphics.tsa.plot_acf(data_stationarized["close"], lags=50)
    # y limit to 0.25
    pyplot.ylim(-0.25, 0.25)
    save_plot(filename="eda/autocorrelation_stationarized_data.png")
    pyplot.show()

    # Partial autocorrelation plot
    sm.graphics.tsa.plot_pacf(data_stationarized["close"], lags=50)
    # y limit to 0.25
    pyplot.ylim(-0.25, 0.25)
    save_plot(filename="eda/partial_autocorrelation_stationarized_data.png")
    pyplot.show()

    logger.info(
        "The autocorrelation plot doesn't show any significant lag for the autocorrelation and the partial "
        "autocorrelation."
    )

    # Plot close prices 2023 differentiated by a week
    data_weekly = data.diff(periods=7).dropna()
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.plot(
        data_weekly["2023-01-01":"2024-01-01"].index,
        data_weekly["2023-01-01":"2024-01-01"]["close"],
        label="Diffenciated Close Price",
    )
    ax.set_title("Diffenciated Close Price Over 2023")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    save_plot(filename="eda/diffenciated_by_week_close_price.png")
    pyplot.show()


def exploratory_data_analysis(data: pandas.DataFrame) -> None:
    """Exploratory data analysis.

    Args:
        data (pandas.DataFrame): data
    """
    # Set pandas options
    pandas.set_option("display.max_columns", None)

    msg_task(msg="Exploratory Data Analysis", logger=logger)

    logger.info(
        "For the analysis, the companies are aggregated by the mean to get a sense of how the index performs as a"
        " whole.\n\n"
    )

    figsize = (14, 8)
    data_analysis(data=data, figsize=figsize)
    time_series_analysis(data=data, figsize=figsize)


if __name__ == "__main__":
    exploratory_data_analysis(data=get_dataset(group_by_date=True))
