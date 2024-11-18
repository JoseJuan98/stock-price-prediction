import logging  
import numpy as np  
import json  
import pandas  

from keras.models import Sequential  
from keras.layers import Input, LSTM, Dense  
from keras.optimizers import Adam  
from sklearn.preprocessing import MinMaxScaler  
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error  
from matplotlib import pyplot  

from common.base_model import BaseModel, ARRAY_LIKE  
from common.data_preparation import get_features_and_target  
from common import get_logger  
from common.config import Config  
  
class LSTMModel(BaseModel):  
    """LSTM model for time series forecasting."""  
  
    def __init__(  
        self,  
        name: str,  
        y_train: ARRAY_LIKE,  
        y_test: ARRAY_LIKE,  
        x_train: ARRAY_LIKE,  
        x_test: ARRAY_LIKE,  
        n_lag: int,  
        n_features: int = 1,  
        lstm_units: int = 50,  
        activation: str = 'relu',  
        epochs: int = 25,  
        batch_size: int = 32,  
        logger: logging.Logger = None,  
    ) -> None:  
        """Initialize the LSTM model with configurable parameters."""  
        self.lstm_units = lstm_units  
        self.activation = activation  
        self.epochs = epochs  
        self.batch_size = batch_size  
        model = self.build_model(n_lag, n_features)  
        super().__init__(  
            model=model,  
            name=name,  
            y_train=y_train,  
            y_test=y_test,  
            x_train=x_train,  
            x_test=x_test,  
            logger=logger  
        )  
        self.n_lag = n_lag  
        self.n_features = n_features  
        self.x_scaler = MinMaxScaler(feature_range=(0, 1))  
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))  
  
    def build_model(self, n_lag, n_features):  
        model = Sequential()  
        model.add(Input(shape=(n_lag, n_features))) 
        model.add(LSTM(self.lstm_units, activation=self.activation))  
        model.add(Dense(1))  
        model.compile(optimizer=Adam(), loss='mse')  
        return model  
  
    def fit(self, force_fit: bool = False) -> None:  
        """Fit the model to the data."""  

        # Scale the data  
        x_train_scaled = self.x_scaler.fit_transform(self.x_train.values)  
        y_train_array = np.array(self.y_train).reshape(-1, 1)  
        y_train_scaled = self.y_scaler.fit_transform(y_train_array) 

        if self.model_file.exists() and not force_fit:  
            self.logger.info(f"Loading model from {self.model_file}")  
            self.load_model()  
        else:  
            self.logger.info("Fitting the model ...")   
  
            # Create the dataset  
            X_train, y_train = self.create_dataset(x_train_scaled, y_train_scaled, self.n_lag)  
  
            # Fit the model  
            self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)  
            self.save_model()  
  
    def create_dataset(self, x_data, y_data, n_lag):  
        X, y = [], []  
        for i in range(len(x_data) - n_lag):  
            X.append(x_data[i:(i + n_lag), :])  
            y.append(y_data[i + n_lag, 0])  
        return np.array(X), np.array(y)   
  
    def forecast(self, steps: int, data: ARRAY_LIKE = None) -> np.ndarray:  
        """Make predictions using the model."""  
        if data is None:  
            # Use the last part of the training data to seed the prediction  
            last_input = self.x_train[-self.n_lag:].reshape(1, self.n_lag, self.n_features)  
            predictions = []  
            current_input = last_input  
    
            for _ in range(steps):  
                pred = self.model.predict(current_input)  
                predictions.append(pred[0, 0])  
                # Update the current input by appending the prediction and discarding the oldest input  
                current_input = np.roll(current_input, shift=-1, axis=1)  
                current_input[0, -1, 0] = pred  
    
            predictions = np.array(predictions).reshape(-1, 1)  
        else:  
            # If data is provided, predict directly  
            x_data_array = np.array(data).reshape(-1, self.n_features)  
            x_data_scaled = self.x_scaler.transform(x_data_array)  
            X_input, _ = self.create_dataset(x_data_scaled, np.zeros((x_data_scaled.shape[0], 1)), self.n_lag)  
            predictions = self.model.predict(X_input)  
    
        # Inverse transform predictions  
        return self.y_scaler.inverse_transform(predictions).flatten()  
  
    def evaluate(self) -> None:  
        """Evaluate the model on the data and store the metrics in `self.metrics`."""  
        if self.x_test is not None and not self.x_test.empty:  
            y_pred = self.forecast(steps=self.y_test.shape[0] - self.n_lag, data=self.x_test)  
        else:  
            raise ValueError("x_test cannot be None or empty for evaluation.")  

        # Slice y_test to match y_pred length  
        y_test_aligned = self.y_test[self.n_lag:]  

        if y_pred.shape[0] != y_test_aligned.shape[0]:  
            raise ValueError(f"Predictions length {y_pred.shape[0]} does not match y_test length {y_test_aligned.shape[0]}.")  

        # Compute metrics  
        self.metrics = {  
            "MAE": round(float(mean_absolute_error(y_true=y_test_aligned, y_pred=y_pred)), 4),  
            "MSE": round(float(root_mean_squared_error(y_true=y_test_aligned, y_pred=y_pred) ** 2), 4),  
            "RMSE": round(float(root_mean_squared_error(y_true=y_test_aligned, y_pred=y_pred)), 4),  
            "MAPE": round(float(mean_absolute_percentage_error(y_true=y_test_aligned, y_pred=y_pred)), 4),  
        } 

        # Save the metrics to the `artifacts/model` directory  
        Config.model_dir.mkdir(parents=True, exist_ok=True)  
        with open(Config.model_dir / f"{self.name}_metrics.json", "w") as file:  
            json.dump(obj=self.metrics, fp=file, indent=4)  

        # Plot the forecasted values  
        self.plot_forecast(y_test_lag_adjusted=y_test_aligned, y_pred=y_pred) 

    def plot_forecast(self, y_test_lag_adjusted, y_pred: np.ndarray) -> None:
        """Plot the forecasted values."""
        y_pred = pandas.Series(data=y_pred.astype(np.float64), index=y_test_lag_adjusted.index)

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
  
def train_and_evaluate():  
    """Train and evaluate the LSTM model."""  
    logger = get_logger(log_filename="lstm.log")  
    X_train, y_train, X_test, y_test = get_features_and_target()  
  
    n_lag = 5  
    n_features = X_train.shape[1]  
      
    # Define model parameters here  
    lstm_units = 50  
    activation = 'relu'  
    epochs = 12  
    batch_size = 32  
  
    model = LSTMModel(  
        name="lstm",  
        y_train=y_train,  
        y_test=y_test,  
        x_train=X_train,  
        x_test=X_test,  
        n_lag=n_lag,  
        n_features=n_features,  
        lstm_units=lstm_units,  
        activation=activation,  
        epochs=epochs,  
        batch_size=batch_size,  
        logger=logger  
    )  
  
    model.fit(force_fit=True)  
    model.evaluate()  
    logger.info(f"Metrics: {model.metrics}")  
  
if __name__ == "__main__":  
    train_and_evaluate()  