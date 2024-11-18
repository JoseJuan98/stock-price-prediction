# -*- coding: utf-8 -*-
"""Utility module.

This module contains the utility functions to be used in the project.
"""
from urllib.request import urlretrieve

import pandas
from matplotlib import pyplot

from common.config import Config


def get_dataset(group_by_date: bool = True) -> pandas.DataFrame:
    """Get data for the project.

    Notes:
        First it checks if the data is already downloaded. If not, it downloads the data.
        Then it loads the data into a pandas DataFrame.

    Args:
        group_by_date (bool): whether to aggregate the data by date or not, to eliminate the symbol (company) column.

    Returns:
        pandas.DataFrame: data
    """
    data_path = Config.data_dir / "daily_sp500.parquet"

    if not data_path.exists():

        # Create folder if it doesn't exist
        data_path.parent.mkdir(parents=True, exist_ok=True)

        # Download data
        urlretrieve(url=Config.data_url, filename=data_path)

    # Load data
    data = pandas.read_parquet(path=data_path, dtype_backend="pyarrow").sort_values("date").set_index(keys="date")
    data.index = pandas.to_datetime(data.index)

    if group_by_date:
        data = data.drop(columns=["symbol"]).groupby(by=["date"]).mean()

    return data


def save_plot(filename: str) -> None:
    """Private method to save the plots"""
    plot_path = Config.plot_dir / filename

    # make dir if it doesn't exist yet
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    pyplot.savefig(plot_path, bbox_inches="tight")
