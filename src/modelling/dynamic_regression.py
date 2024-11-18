# -*- utf-8 -*-
"""Dynamic Regression model for time series forecasting."""

import logging

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

from common import get_logger
from common.base_model import BaseModel, ARRAY_LIKE
from common.data_preparation import get_features_and_target


class DynamicRegression(BaseModel):
    """Dynamic Regression model for time series forecasting."""

    def __init__(
        self,
        name: str,
        y_train: ARRAY_LIKE,
        y_test: ARRAY_LIKE,
        x_train: ARRAY_LIKE,
        x_test: ARRAY_LIKE,
        order=(1, 0, 1),
        seasonal_order=(0, 0, 0, 0),
        logger: logging.Logger = None,
    ) -> None:
        """Initialize the Dynamic Regression model.

        Args:
            name (str): The name of the model.
            y_train (ARRAY_LIKE): The target variable for training the model.
            y_test (ARRAY_LIKE): The target variable for testing the model.
            x_train (ARRAY_LIKE): The features for training the model.
            x_test (ARRAY_LIKE): The features for testing the model.
            order (tuple): The (p, d, q) order of the model for the number of AR parameters, differences, and MA parameters.
            seasonal_order (tuple): The (P, D, Q, s) seasonal order of the model.
            logger (logging.Logger): The logger object.
        """
        super().__init__(
            model=SARIMAX(
                endog=y_train.to_numpy(),
                exog=x_train.to_numpy(),
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ),
            name=name,
            y_train=y_train,
            y_test=y_test,
            x_train=x_train,
            x_test=x_test,
            logger=logger,
        )
        self.order = order
        self.seasonal_order = seasonal_order
        self.model: SARIMAX | SARIMAXResults

    def fit(self, force_fit: bool = False) -> None:
        """Fit the model to the data."""
        if self.model_file.exists() and not force_fit:
            self.logger.info(f"Loading model from {self.model_file}")
            self.load_model()
        else:
            self.logger.info("Fitting the model ...")
            self.model = self.model.fit(disp=False)
            self.save_model()

    def forecast(self, steps: int, data: ARRAY_LIKE = None) -> np.ndarray:
        """Make predictions using the model."""
        self.model: SARIMAXResults
        return self.model.forecast(steps=steps, exog=data)


def train_and_evaluate():
    """Train and evaluate the Dynamic Regression model."""
    logger = get_logger(log_filename="dynamic_regression.log")
    X_train, y_train, X_test, y_test = get_features_and_target()

    logger.info("Training the Dynamic Regression model ...")

    cols = ["trade_value_rsi", "trade_value_bb_upper", "trade_value_bb_lower"]

    # Initialize the model
    dr_model = DynamicRegression(
        name="dynamic_regression",
        x_train=X_train[cols],
        y_train=y_train,
        x_test=X_test[cols],
        y_test=y_test,
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 0),
        logger=logger,
    )

    # Fit the model
    dr_model.fit(force_fit=False)

    # Evaluate the model
    dr_model.evaluate()

    # Print the metrics
    logger.info(f"Best summary: {dr_model.model.summary()}")
    logger.info(f"Metrics: {dr_model.metrics}")


if __name__ == "__main__":
    train_and_evaluate()
