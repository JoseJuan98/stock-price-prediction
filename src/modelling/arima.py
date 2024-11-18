# -*- utf-8 -*-
"""ARIMA model for time series forecasting.

References:
    - https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_forecasting.html
"""
import logging

import numpy
import pandas
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA as stats_ARIMA, ARIMAResults

from common import get_logger, save_plot
from common.base_model import BaseModel, ARRAY_LIKE
from common.config import Config
from common.data_preparation import get_features_and_target


class ARIMA(BaseModel):
    """ARIMA model for time series forecasting."""

    def __init__(
        self,
        name: str,
        y_train: ARRAY_LIKE,
        y_test: ARRAY_LIKE,
        order: tuple[int, int, int],
        logger: logging.Logger = None,
    ) -> None:
        """Initialize the ARIMA model.

        Args:
            name (str): The name of the model.
            y_train (ARRAY_LIKE): The target variable for training the model.
            y_test (ARRAY_LIKE): The target variable for testing the model.
            order (tuple[int, int, int]): The order (p,d,q) of the ARIMA model.
            logger (logging.Logger): The logger object.
        """
        super().__init__(
            model=stats_ARIMA(endog=y_train.to_numpy(), order=order),
            name=name,
            y_train=y_train,
            y_test=y_test,
            logger=logger,
        )
        self.order = order
        self.model: stats_ARIMA | ARIMAResults

    def fit(self, force_fit: bool = False) -> None:
        """Fit the model to the data.

        Args:
            force_fit (bool): Whether to force the fitting of the model. By default, False.
        """
        if self.model_file.exists() and not force_fit:
            self.logger.info(f"Loading model from {self.model_file}")
            self.load_model()
        else:
            self.logger.info("Fitting the model ...")

            # Fit the model
            self.model: stats_ARIMA
            self.model = self.model.fit()

            # Save the model
            self.save_model()

    def forecast(self, steps: int, data: ARRAY_LIKE = None) -> numpy.ndarray:
        """Make predictions using the model.

        Args:
            data (ARRAY_LIKE): The data to make predictions on.
            steps (int): The number of steps to forecast.

        Returns:
            numpy.ndarray: The forecasted values.
        """
        return self.model.forecast(steps=steps)


def train_and_evaluate():
    """Train and evaluate the ARIMA model."""
    logger = get_logger(log_filename="arima.log")
    _, y_train, _, y_test = get_features_and_target()

    # Order is the main parameter of the ARIMA model
    # p: The number of lag observations included in the model (AR)
    # d: The degree of differencing (I)
    # q: The size of the moving average window (MA)
    # order = (p, d, q)
    orders = [(1, 1, 2), (1, 1, 1), (2, 1, 1), (2, 1, 2)]

    best_arima = None
    best_mape = float("inf")

    # try the better orders based on the feature engineering analysis
    for order in orders:

        logger.info(f"\nFitting ARIMA model with order: {order}\n")

        # Initialize the ARIMA model
        arima = ARIMA(name=f"arima_{order[0]}_{order[1]}_{order[2]}", y_train=y_train, y_test=y_test, order=order)

        # Fit the model
        arima.fit(force_fit=False)

        # Evaluate the model
        arima.evaluate()

        # Print the metrics
        print(arima.metrics)

        if arima.metrics["MAPE"] < best_mape:
            best_mape = arima.metrics["MAPE"]
            best_arima = arima

    logger.info(f"Best order: {best_arima.order}\n With metrics: {best_arima.metrics}")
    logger.info(f"Model summary: {best_arima.model.summary()}")

    # Plot the residuals
    residuals = pandas.DataFrame(data=best_arima.model.resid, columns=["Residuals"])
    residuals.plot(kind="kde")
    pyplot.title(f"Residuals of {best_arima.name}")
    save_plot(Config.plot_dir / "trade_value" / f"residuals_{best_arima.name}.png")
    pyplot.show()


if __name__ == "__main__":
    train_and_evaluate()
