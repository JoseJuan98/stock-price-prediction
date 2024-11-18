# -*- utf-8 -*-
"""Gradient Boosting Machine model for time series forecasting."""

import logging

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from common import get_logger
from common.base_model import BaseModel, ARRAY_LIKE
from common.data_preparation import get_features_and_target


class GBM(BaseModel):
    """Gradient Boosting Machine model for time series forecasting."""

    def __init__(
        self,
        name: str,
        y_train: ARRAY_LIKE,
        y_test: ARRAY_LIKE,
        x_train: ARRAY_LIKE,
        x_test: ARRAY_LIKE,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=None,
        logger: logging.Logger = None,
    ) -> None:
        """Initialize the GBM model.

        Args:
            name (str): The name of the model.
            y_train (ARRAY_LIKE): The target variable for training the model.
            y_test (ARRAY_LIKE): The target variable for testing the model.
            x_train (ARRAY_LIKE): The features for training the model.
            x_test (ARRAY_LIKE): The features for testing the model.
            n_estimators (int): The number of boosting stages to perform.
            learning_rate (float): Learning rate shrinks the contribution of each tree.
            max_depth (int): Maximum depth of the individual regression estimators.
            random_state (int): The seed of the pseudo random number generator.
            logger (logging.Logger): The logger object.
        """
        super().__init__(
            model=GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
                verbose=1,
            ),
            name=name,
            y_train=y_train,
            y_test=y_test,
            x_train=x_train,
            x_test=x_test,
            logger=logger,
        )

    def fit(self, force_fit: bool = False) -> None:
        """Fit the model to the data."""
        if self.model_file.exists() and not force_fit:
            self.logger.info(f"Loading model from {self.model_file}")
            self.load_model()
        else:

            self.logger.info("Fitting the GBM model ...")
            self.model.fit(self.x_train, self.y_train)
            self.save_model()

    def forecast(self, steps: int = None, data: ARRAY_LIKE = None) -> np.ndarray:
        """Make predictions using the model."""
        return self.model.predict(data)


def train_and_evaluate():
    """Train and evaluate the GBM model."""
    logger = get_logger(log_filename="gbm.log")
    X_train, y_train, X_test, y_test = get_features_and_target()

    cols = ["trade_value_rsi", "trade_value_bb_upper", "trade_value_bb_lower"]

    # Initialize the model
    gbm_model = GBM(
        name="GBM_Model",
        x_train=X_train[cols],
        y_train=y_train,
        x_test=X_test[cols],
        y_test=y_test,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        logger=logger,
    )

    # Fit the model
    gbm_model.fit(force_fit=True)

    # Evaluate the model
    gbm_model.evaluate()

    logger.info("Training and evaluation completed.")
    logger.info(f"Model metrics: {gbm_model.metrics}")


if __name__ == "__main__":
    train_and_evaluate()
