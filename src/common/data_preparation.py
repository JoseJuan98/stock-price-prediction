# -*- coding: utf-8 -*-
"""Data Preparation module.

This module contains the functions for data preparation.
"""

import joblib
import pandas

from common import get_dataset, get_logger
from common.config import Config
from common.feature_engineering import add_feature


def prepare_data_for_training(data: pandas.DataFrame) -> pandas.DataFrame:
    """Prepare data for training.

    Args:
        data (pandas.DataFrame): The input DataFrame

    Returns:
        pandas.DataFrame: The prepared data
    """
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

    # Drop rows with missing values
    data = data.dropna()

    # Drop duplicate rows
    data = data.drop_duplicates()

    return data


def get_features_and_target(
    force_split: bool = False,
) -> (pandas.DataFrame, pandas.Series, pandas.DataFrame, pandas.Series):
    """Get data for training.

    Notes:
        If the data is already split and stored in the disk, it will be loaded from there. Otherwise, if it's the
        first time using the function or force_split=True the data will be split and stored in the disk.

    Args:
        force_split (bool): Whether to force the split of the data and store the new results in the disk. By default,
                            False.

    Returns:
        pandas.DataFrame: The training features
        pandas.Series: The training target
        pandas.DataFrame: The test features
        pandas.Series: The test target
    """
    prepared_data_path = Config.data_dir / "prepared_data.joblib"

    if force_split or not prepared_data_path.exists():
        logger = get_logger()

        logger.info("Data preparation started /> ...")
        data = get_dataset(group_by_date=True)

        data = prepare_data_for_training(data)

        # Last day of dataset: 2024-02-26
        # Split data into training and test sets
        train = data[:"2024-01-01"]
        test = data["2024-01-02":]

        # Split data into features and target
        X_train = train.drop(columns=["trade_value", "close"])
        y_train = train["trade_value"]
        X_test = test.drop(columns=["trade_value", "close"])
        y_test = test["trade_value"]

        joblib.dump(
            value={
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
            },
            filename=prepared_data_path,
        )
        logger.info("/> Data preparation completed.")
    else:
        data = joblib.load(filename=prepared_data_path)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]

    return X_train, y_train, X_test, y_test
