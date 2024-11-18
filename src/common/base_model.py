# -*- utf-8 -*-
"""Base class for all models."""

import json
import logging
from abc import ABC
from typing import Any, Union

import joblib
import numpy
import pandas
from matplotlib import pyplot
from sklearn.base import BaseEstimator as sklearn_model
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)
from statsmodels.base.model import Model as stats_model
from torch._tensor import Tensor
from torch.nn import Module as torch_model

from common import get_logger
from common.config import Config

ARRAY_LIKE = Union[pandas.DataFrame, pandas.Series, numpy.ndarray, Tensor]


class BaseModel(ABC):
    """Base class for all models."""

    def __init__(
        self,
        model: sklearn_model | stats_model | torch_model | Any,
        name: str,
        y_train: ARRAY_LIKE,
        y_test: ARRAY_LIKE,
        x_train: ARRAY_LIKE = None,
        x_test: ARRAY_LIKE = None,
        logger: logging.Logger = None,
    ):
        """Initialize the base model.

        Args:
            model (sklearn.base.BaseEstimator | statsmodels.base.model.Model | torch.nn.Module):
                The model to be used for forecasting.
            name (str): The name of the model.
            x_train (ARRAY_LIKE): The features for training the model.
            y_train (ARRAY_LIKE): The target variable for training the model.
            x_test (ARRAY_LIKE): The features for testing the model.
            y_test (ARRAY_LIKE): The target variable for testing the model.
            logger (logging.Logger): The logger object.
        """
        self.model = model
        self.name = name
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.metrics: dict[str, float] = {}
        self.model_file = Config.model_dir / f"{self.name}.joblib"

        self.logger = logger or get_logger(name=self.name)

    def save_model(self) -> None:
        """Save the model to the `artifacts/model` directory."""
        self.logger.info(f"Saving model to {self.model_file}")

        # Create the `artifacts/model` directory if it doesn't exist
        self.model_file.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(value=self.model, filename=self.model_file)

    def load_model(self) -> None:
        """Load the model from the `artifacts/model` directory."""
        self.model = joblib.load(filename=self.model_file)

    def fit(self, force_fit: bool = False) -> None:
        """Fit the model to the data.

        Args:
            force_fit (bool): Whether to force the fitting of the model. By default, False.
        """
        raise NotImplementedError

    def forecast(self, steps: int, data: ARRAY_LIKE = None) -> numpy.ndarray:
        """Make predictions using the model.

        Args:
            data (ARRAY_LIKE): The data to make predictions on.
            steps (int): The number of steps to forecast.

        Returns:
            numpy.ndarray: The forecasted values.
        """
        raise NotImplementedError

    def evaluate(self) -> None:
        """Evaluate the model on the data and store the metrics in `self.metrics` and in the
        `artifacts/model/<self.name>_metrics.json` file.

        Notes:
            This method calculates the following metrics:
                - Mean Absolute Error (MAE)
                - Mean Squared Error (MSE)
                - Root Mean Squared Error (RMSE)
                - Mean Absolute Percentage Error (MAPE)
        """
        # For models that require features, e.g., VAR, LSTM
        if self.x_test is not None:
            y_pred = self.forecast(data=self.x_test, steps=self.y_test.shape[0])
        # For models that don't require features, e.g., ARIMA
        else:
            y_pred = self.forecast(steps=self.y_test.shape[0])

        self.metrics = {
            "MAE": round(float(mean_absolute_error(y_true=self.y_test, y_pred=y_pred)), 4),
            "MSE": round(float(mean_squared_error(y_true=self.y_test, y_pred=y_pred)), 4),
            "RMSE": round(float(root_mean_squared_error(y_true=self.y_test, y_pred=y_pred)), 4),
            "MAPE": round(float(mean_absolute_percentage_error(y_true=self.y_test, y_pred=y_pred)), 4),
        }

        # Save the metrics to the `artifacts/model` directory
        Config.model_dir.mkdir(parents=True, exist_ok=True)

        with open(Config.model_dir / f"{self.name}_metrics.json", "w") as file:
            json.dump(obj=self.metrics, fp=file, indent=4)

        assert y_pred.shape[0] == self.y_test.shape[0], "Forecasted values and test values should have the same length."

        # Plot the forecasted values
        self.plot_forecast(y_pred=y_pred)

    def plot_forecast(self, y_pred: numpy.ndarray) -> None:
        """Plot the forecasted values."""
        y_pred = pandas.Series(data=y_pred.astype(numpy.float64), index=self.y_test.index)

        plot_file = Config.plot_dir / "forecast" / f"{self.name}_forecast.png"
        plot_file.parent.mkdir(parents=True, exist_ok=True)

        # Smooth the plot as it's cut off in a weekend
        y_train = self.y_train.copy()
        y_train.loc[self.y_test.index[0]] = self.y_test[0]

        # too similar y_test and y_pred to see the difference
        o_marker = self.metrics["MAPE"] < 0.001

        pyplot.figure(figsize=(19, 6))
        pyplot.plot(y_train["2023-08-01":], label="Train", color="blue")
        pyplot.plot(self.y_test, label="Test", color="green", marker="o" if o_marker else None)
        pyplot.plot(y_pred, label="Forecast", color="orange")
        pyplot.legend(loc="best")
        pyplot.title(f"{self.name} Forecast")
        pyplot.xlabel("Time")
        pyplot.ylabel("Value")
        pyplot.savefig(plot_file)
        pyplot.show()
