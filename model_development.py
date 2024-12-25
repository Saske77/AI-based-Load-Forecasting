import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import lightgbm as lgb


class EnergyLoadModels:
    def __init__(self):
        self.models = {}

    def baseline_linear_regression(self, X_train, X_test, y_train, y_test):
        """Linear Regression baseline model"""
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        self.models['linear_regression'] = {
            'model': lr_model,
            'mae': mae,
            'rmse': rmse
        }
        return mae, rmse

    def arima_model(self, energy_series):
        """ARIMA time series forecasting"""
        model = ARIMA(energy_series, order=(5, 1, 2))
        model_fit = model.fit()

        self.models['arima'] = model_fit
        return model_fit

    def lstm_network(self, X_train, y_train, X_test, y_test):
        """LSTM Neural Network for time series"""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(X_train.shape[1], 1)))
        model.add(tf.keras.layers.LSTM(50, activation='relu'))
        model.add(tf.keras.layers.Dense(1))

        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        self.models['lstm'] = {
            'model': model,
            'mae': mae,
            'rmse': rmse
        }
        return mae, rmse

    def xgboost_model(self, X_train, X_test, y_train, y_test):
        """XGBoost Model"""
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        self.models['xgboost'] = {
            'model': model,
            'mae': mae,
            'rmse': rmse
        }
        return mae, rmse

    def compare_models(self):
        """Compare performance of different models"""
        performance_df = pd.DataFrame([
            {name: model.get('mae', None) for name, model in self.models.items()},
            {name: model.get('rmse', None) for name, model in self.models.items()}
        ], index=['MAE', 'RMSE'])

        return performance_df