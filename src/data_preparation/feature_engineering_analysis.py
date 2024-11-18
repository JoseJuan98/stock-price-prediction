# -*- coding: utf-8 -*-
"""Feature Engineering Analysis module.

This module contains the analysis of the feature engineering process.
"""

import numpy
import pandas
import seaborn
from matplotlib import pyplot

from common import get_dataset, get_logger, msg_task, save_plot, calculate_vif
from common.feature_engineering import add_feature


def feature_engineering_analysis(data: pandas.DataFrame) -> None:
    """Feature engineering analysis."""
    logger = get_logger(log_filename="feature_engineering_analysis.log")

    logger.info("Feature engineering analysis started.")
    logger.info(f"Data: {data.head()}")

    # Add Stock Trade Value as target variable
    data = add_feature(feature="Stock_Trade_Value", data=data)

    # Add SMA
    data = add_feature(feature="SMA", data=data, cols="trade_value", window=1)

    # Add RSI
    data = add_feature(feature="RSI", data=data, cols="trade_value", window=1)

    # Add Bollinger Bands
    data = add_feature(feature="Bollinger_Bands", data=data, cols="trade_value", window=7, num_std_dev=2)

    # Add Lag features
    data = add_feature(feature="Lag", data=data, cols="trade_value", lags=[1, 7])

    # Add Volatility
    data = add_feature(feature="Volatility", data=data)

    msg_task("Feature engineering analysis completed.", logger=logger)

    # Plot each column
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(16, 9))
    data_2023 = data["2023-01-01":"2024-01-01"].copy()
    for col in [col for col in data.columns.tolist() if "trade_value_" in col and "rsi" not in col]:
        ax.plot(data_2023.index, numpy.log(data_2023[col]), label=col)
    ax.set_xlabel("Date")
    ax.set_ylabel("Feature Values (log)")
    ax.plot(
        data_2023.index,
        numpy.log(data_2023["trade_value"].values),
        label="trade_value",
        marker="o",
        color="black",
    )
    ax.legend()
    pyplot.gcf().autofmt_xdate()
    pyplot.title("Trade Value Features Over 2023")
    save_plot(filename="trade_value_features_log_over_2023.png")
    pyplot.show()

    # Correlation heatmap
    pyplot.figure(figsize=(16, 9))
    seaborn.heatmap(data.corr(), annot=True, cmap="coolwarm")
    pyplot.title("Correlation Matrix")
    pyplot.tight_layout()
    pyplot.gcf().autofmt_xdate()
    save_plot(filename="features_correlation_matrix.png")
    pyplot.show()

    # Calculate VIF
    # Normalize the features before analysis
    normalized_data = data.drop(columns=["trade_value"]).dropna()
    normalized_data = (normalized_data - normalized_data.mean()) / normalized_data.std()
    vif = calculate_vif(data=normalized_data)
    logger.info(f"\n\nVIF values:\n\n{vif}\n\n")

    # The value to be considered as high collinearity is relative and depends on the context of the problem.
    # In this case, we consider a VIF value greater than 35 as high collinearity.
    high_vif = vif[vif > 35]
    if not high_vif.empty:
        logger.info(
            f"\n\nFor columns {vif.index.tolist()}\n\nColumns with high VIF values, thus high collinearity:"
            f"\n\n{high_vif}\n\n"
        )
    else:
        logger.info("No columns with high collinearity.")

    logger.info(
        "Some features, like RSI or Bollinger Bands, might capture aspects of the data that are not evident from the\n"
        " raw time series alone, such as momentum and volatility. These features can potentially improve the model's\n"
        " performance, while moving averages and their lags, can be highly collinear with each other and with the\n"
        " original time series data. High collinearity can lead to unstable coefficient estimates in VAR models, \n"
        "making the model sensitive to small changes in the data and potentially leading to incorrect inferences\n"
        " about the relationships between variables."
    )


if __name__ == "__main__":
    feature_engineering_analysis(data=get_dataset(group_by_date=True))
