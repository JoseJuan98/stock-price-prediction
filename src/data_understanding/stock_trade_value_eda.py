# -*- coding: utf-8 -*-
"""Exploratory Data Analysis.

This module contains the exploratory data analysis utilities to be used in the project.
"""

import pandas
import seaborn
import statsmodels.api as sm
from matplotlib import pyplot

from common import get_dataset, get_logger, msg_task, save_plot, check_stationarity
from common.feature_engineering import add_feature

logger = get_logger()


def trade_value_analysis(data: pandas.DataFrame, figsize: tuple = (16, 9)) -> None:
    """Data analysis.

    Args:
        data (pandas.DataFrame): data
        figsize (set, optional): figure size. Defaults to (14, 8).
    """
    # Plotting
    logger.info("Data Visualization\n\n")

    # Plot trade_value prices
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.plot(
        data.index,
        data["trade_value"],
        label="trade_value Price",
    )
    ax.set_title("Stock Trade Value Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("trade_value Price")
    save_plot(filename="trade_value/trade_value_price_over_time.png")
    pyplot.show()

    # Plot all prices in 2023
    var_color = {
        "trade_value": "blue",
    }
    data_23 = data["2023-01-01":"2024-01-01"]
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=figsize)
    for var, color in var_color.items():
        ax.plot(data_23.index, data_23[var], label=var.title(), color=color)

    ax.set_title("Price Over 2023")
    ax.set_xlabel("Date")
    pyplot.legend()
    save_plot(filename="trade_value/price_over_2023.png")
    pyplot.show()

    # Correlation heatmap
    seaborn.heatmap(data.corr(), annot=True, cmap="coolwarm")
    pyplot.title("Correlation Matrix with Stock Trade Value")
    pyplot.gcf().autofmt_xdate()
    pyplot.tight_layout()
    save_plot(filename="trade_value/correlation_matrix.png")
    pyplot.show()

    # Boxplot for price distribution
    seaborn.boxplot(data=data["trade_value"])
    pyplot.title("Stock Trade Value Distribution")
    pyplot.tight_layout()
    save_plot(filename="trade_value/price_distribution.png")
    pyplot.show()


def time_series_analysis(data: pandas.DataFrame, figsize: tuple = (14, 8)) -> None:
    """Time series analysis.

    Args:
        data (pandas.DataFrame): data
        figsize (set, optional): figure size. Defaults to (14, 8).
    """

    # Plot monthly trade_value prices to check for long-term trends
    data_monthly = data.resample("ME").mean()
    rolling_avg_12_months = data_monthly.rolling(window=12).mean()
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=figsize)
    period = slice("2001", "2024")
    data_monthly[period]["trade_value"].plot(
        ax=ax,
        label="Montly trade_value Price",
        marker=".",
    )
    rolling_avg_12_months[period]["trade_value"].plot(
        label="Long-term Trend (12 Months Rolling Avg)",
        color="red",
    )
    ax.set_title("Monthly Stock Trade Value Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Trade Value")
    pyplot.legend()
    save_plot(filename="trade_value/monthly_trade_value_price.png")
    pyplot.show()

    logger.info("The plot shows a long-term trend in the trade_value price, and a possible yearly seasonality.\n\n")

    # Seasonal decomposition
    data_monthly.diff(12)[period]["trade_value"].plot(
        marker=".", figsize=(16, 9), label="Yearly Seasonal Decomposition"
    )
    pyplot.title("Diffenciated Yearly trade_value Price")
    pyplot.xlabel("Date")
    pyplot.ylabel("trade_value Price")
    pyplot.gcf().autofmt_xdate()
    save_plot(filename="trade_value/yearly_differentiated_trade_value_price.png")
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

    # Plot trade_value prices differentiated
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.plot(
        data_stationarized.index,
        data_stationarized["trade_value"],
        label="Diffenciated Sotck Trade Value",
    )
    ax.set_title("Diffenciated trade_value Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Trade Value Price")
    save_plot(filename="trade_value/diffenciated_trade_value_price_over_time.png")
    pyplot.show()

    # Plot trade_value with its first and second order differences
    fig, ax = pyplot.subplots(nrows=3, ncols=1, figsize=figsize)
    start_date = "2020-01-01"
    ax[0].plot(
        data[start_date:].index,
        data[start_date:]["trade_value"],
        label="Stock Trade Value",
    )
    ax[0].set_title("Stock Trade Value Series")
    ax[0].axes.xaxis.set_visible(False)
    ax[0].set_ylabel("Stock Trade Value")
    ax[1].plot(
        data_stationarized[start_date:].index,
        data_stationarized[start_date:]["trade_value"],
        label="First Order Difference",
    )
    ax[1].set_title("First Order Difference")
    ax[1].axes.xaxis.set_visible(False)
    ax[1].set_ylabel("Stock Trade Value")
    ax[2].plot(
        data_stationarized[start_date:]["trade_value"].diff(periods=1).dropna().index,
        data_stationarized[start_date:]["trade_value"].diff(periods=1).dropna(),
        label="Second Order Difference",
    )
    ax[2].set_title("Second Order Difference")
    ax[2].set_xlabel("Date")
    ax[2].set_ylabel("Stock Trade Value")
    pyplot.gcf().autofmt_xdate()
    pyplot.tight_layout()
    save_plot(filename="trade_value/first_and_second_order_difference_trade_value.png")
    pyplot.show()

    # Plot trade_value prices differentiated 2023
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.plot(
        data_stationarized["2023-01-01":"2024-01-01"].index,
        data_stationarized["2023-01-01":"2024-01-01"]["trade_value"],
        label="Diffenciated trade_value Price",
    )
    ax.set_title("Diffenciated trade_value Price Over 2023")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Trade Value")
    save_plot(filename="trade_value/diffenciated_trade_value_price_over_2023.png")
    pyplot.show()

    logger.info("The plot shows a stationary time series. So, now we can check seasonality.\n\n")

    # Autocorrelation plots
    fig, ax = pyplot.subplots(nrows=3, ncols=1, figsize=figsize)
    sm.graphics.tsa.plot_acf(data["trade_value"], lags=50, ax=ax[0])
    sm.graphics.tsa.plot_acf(data_stationarized["trade_value"], lags=50, ax=ax[1])
    sm.graphics.tsa.plot_acf(data_stationarized["trade_value"].diff(periods=1).dropna(), lags=50, ax=ax[2])
    # y limit to 0.25
    # pyplot.ylim(-0.25, 0.25)
    ax[1].axes.xaxis.set_visible(False)
    ax[0].axes.xaxis.set_visible(False)
    save_plot(filename="trade_value/autocorrelation_stationarized_data.png")
    pyplot.show()

    # Partial autocorrelation plots
    fig, ax = pyplot.subplots(nrows=3, ncols=1, figsize=figsize)
    sm.graphics.tsa.plot_pacf(data["trade_value"], lags=50, ax=ax[0])
    sm.graphics.tsa.plot_pacf(data_stationarized["trade_value"], lags=50, ax=ax[1])
    sm.graphics.tsa.plot_pacf(data_stationarized["trade_value"].diff(periods=1).dropna(), lags=50, ax=ax[2])
    ax[1].axes.xaxis.set_visible(False)
    ax[0].axes.xaxis.set_visible(False)

    # y limit to 0.25
    # pyplot.ylim(-0.25, 0.25)
    save_plot(filename="trade_value/partial_autocorrelation_stationarized_data.png")
    pyplot.show()

    logger.info(
        "The autocorrelation plot doesn't show any significant lag for the autocorrelation and the partial "
        "autocorrelation."
    )

    # Plot trade_value prices 2023 differentiated by a week
    data_weekly = data.diff(periods=7).dropna()
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.plot(
        data_weekly["2023-01-01":"2024-01-01"].index,
        data_weekly["2023-01-01":"2024-01-01"]["trade_value"],
        label="Diffenciated Stock Trade Value",
    )
    ax.set_title("Diffenciated trade_value Price Over 2023")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Trade Value")
    save_plot(filename="trade_value/diffenciated_by_week_trade_value_price.png")
    pyplot.show()

    # group by day of the week
    data_stationarized["day_of_week"] = pandas.Categorical(
        values=data_stationarized.index.day_name(),
        categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        ordered=True,
    )
    data_grouped = data_stationarized.groupby("day_of_week").mean()
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.plot(
        data_grouped.index,
        data_grouped["trade_value"],
        label="Stock Trade Value per Day of the Week",
        marker=".",
    )
    pyplot.title("Stock Trade Value per Day of the Week")
    pyplot.xlabel("Day of the Week")
    pyplot.ylabel("Stock Trade Value")
    pyplot.tight_layout()
    save_plot(filename="trade_value/trade_value_per_day_of_the_week.png")
    pyplot.show()

    # group by month of the year
    data_stationarized["month"] = data_stationarized.index.month
    data_grouped = data_stationarized.drop(columns=["day_of_week"]).groupby(by="month").mean().sort_index()
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.plot(
        data_grouped.index,
        data_grouped["trade_value"],
        label="Stock Trade Value per Month",
        marker=".",
    )
    pyplot.title("Stock Trade Value per Month")
    pyplot.xlabel("Month")
    pyplot.ylabel("Stock Trade Value")
    pyplot.tight_layout()
    save_plot(filename="trade_value/trade_value_per_month.png")
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
    trade_value_analysis(data=data, figsize=figsize)
    time_series_analysis(data=data, figsize=figsize)


if __name__ == "__main__":
    data = add_feature(feature="Stock_Trade_Value", data=get_dataset(group_by_date=True))
    exploratory_data_analysis(data=data)
