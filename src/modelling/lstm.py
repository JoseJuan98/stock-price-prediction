# -*- coding: utf-8 -*-
"""LSTM model for time series forecasting."""

from common.base_model import BaseModel
from common.data_preparation import get_features_and_target
from common import get_logger


class LSTMModel(BaseModel):
    """LSTM model for time series forecasting."""

    # TODO:


def train_and_evaluate():
    """Train and evaluate the LSTM model."""
    logger = get_logger(log_filename="lstm.log")
    X_train, y_train, X_test, y_test = get_features_and_target()

    model = LSTMModel(
        name="lstm",
        y_train=y_train,
        y_test=y_test,
        x_train=X_train,
        x_test=X_test,
        logger=logger,
        model=None,
    )

    model.fit(force_fit=True)
    model.evaluate()
    logger.info(f"Metrics: {model.metrics}")


if __name__ == "__main__":
    train_and_evaluate()
