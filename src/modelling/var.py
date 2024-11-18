# -*- utf-8 -*-
"""VAR model for time series forecasting.

References:
    - https://www.statsmodels.org/dev/vector_ar.html
"""
import logging

import joblib
import numpy
import pandas
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.vector_ar.var_model import VAR as stats_VAR, VARResults

from common import get_logger
from common.base_model import BaseModel, ARRAY_LIKE
from common.config import Config
from common.data_preparation import get_features_and_target
from common.stats import __check_stationarity


class VAR(BaseModel):
    """VAR model for time series forecasting."""

    def __init__(
        self,
        name: str,
        y_train: ARRAY_LIKE,
        y_test: ARRAY_LIKE,
        x_train: ARRAY_LIKE,
        x_test: ARRAY_LIKE,
        logger: logging.Logger = None,
    ) -> None:
        """Initialize the ARIMA model.

        Args:
            name (str): The name of the model.
            y_train (ARRAY_LIKE): The target variable for training the model.
            y_test (ARRAY_LIKE): The target variable for testing the model.
            logger (logging.Logger): The logger object.
            x_train (ARRAY_LIKE): The features for training the model.
            x_test (ARRAY_LIKE): The features for testing the model.
        """
        # In the VAR (Vector Autoregression) model, all variables in the system are treated as endogenous. That is,
        # each variable in the system is modeled as a linear combination of past values of itself and past values of
        # all other variables in the system.
        # endog = x_train.join(y_train).dropna()
        super().__init__(
            model=stats_VAR(endog=x_train.to_numpy().astype(dtype=numpy.float64)),
            name=name,
            y_train=y_train,
            y_test=y_test,
            x_train=x_train,
            x_test=x_test,
            logger=logger,
        )
        self.model: stats_VAR | VARResults
        self.lr: LinearRegression = LinearRegression(n_jobs=-1)
        # self.x_train = endog
        # self.y_train = None

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
            self.model: stats_VAR
            self.model = self.model.fit(maxlags=15, ic="aic")
            self.lr.fit(X=self.x_train, y=self.y_train)

            # Save the model
            self.save_model()

    def save_model(self) -> None:
        """Save the model to the `artifacts/model` directory."""
        super().save_model()
        joblib.dump(value=self.lr, filename=Config.model_dir / f"{self.name}_lr.joblib")

    def load_model(self) -> None:
        """Load the model from the `artifacts/model` directory."""
        super().load_model()
        self.lr = joblib.load(filename=Config.model_dir / f"{self.name}_lr.joblib")

    def forecast(self, steps: int, data: ARRAY_LIKE = None) -> numpy.ndarray:
        """Make predictions using the model.

        Args:
            data (ARRAY_LIKE): The data to make predictions on.
            steps (int): The number of steps to forecast.

        Returns:
            numpy.ndarray: The forecasted values.
        """
        forecast_input = data.values[-self.model.k_ar :]
        predicted_x_test = self.model.forecast(y=forecast_input, steps=steps)
        # first oreder diff
        predicted_x_test = pandas.DataFrame(
            data=predicted_x_test, columns=self.x_test.columns, index=self.x_test.index
        ).diff(periods=1)
        predicted_x_test = predicted_x_test.fillna(value=predicted_x_test.mean()).to_numpy()
        return self.lr.predict(X=predicted_x_test)


def train_and_evaluate():
    """Train and evaluate the ARIMA model."""
    logger = get_logger(log_filename="var.log")
    X_train, y_train, X_test, y_test = get_features_and_target()

    cols = ["trade_value_rsi", "trade_value_bb_upper", "trade_value_bb_lower"]

    # Stationarize the time series data
    X_train = X_train[cols].diff(periods=1).dropna()
    X_test = X_test[cols].diff(periods=1).dropna()
    y_train = y_train.diff(periods=1).dropna()
    y_test = y_test.diff(periods=1).dropna()

    assert all(__check_stationarity(X_train[col]) for col in X_train.columns), "X_train is not stationary"
    assert all(__check_stationarity(X_test[col]) for col in X_test.columns), "X_test is not stationary"
    assert __check_stationarity(y_train), "y_train is not stationary"
    assert __check_stationarity(y_test), "y_test is not stationary"

    # Initialize the ARIMA model
    var = VAR(
        name="var_trade_value_lag1_7_diff1",
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
    )

    # Fit the model
    var.fit(force_fit=True)

    # Evaluate the model
    var.evaluate()

    # Print the metrics
    logger.info(f"Model summary: {var.model.summary()}")
    logger.info(f"Metrics: {var.metrics}")


if __name__ == "__main__":
    train_and_evaluate()
