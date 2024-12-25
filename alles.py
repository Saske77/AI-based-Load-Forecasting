import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class EnergyDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def load_energy_data(self, filepath):
        """
        Load energy consumption data from CSV file

        Parameters:
        filepath (str): Path to the CSV file (e.g., 'german_energy_data.csv')

        Returns:
        pd.DataFrame: Loaded and basic preprocessed energy data
        """
        # Load the CSV file
        df = pd.read_csv(filepath)

        # Convert datetime column to pandas datetime
        df['datetime'] = pd.to_datetime(df['datetime'])

        return df

    def engineer_features(self, df):
        """
        Create time-based and weather-related features from the loaded data
        """
        # Create copy to avoid modifying original data
        energy_df = df.copy()

        # Time-based features
        energy_df['hour'] = energy_df['datetime'].dt.hour
        energy_df['day_of_week'] = energy_df['datetime'].dt.dayofweek
        energy_df['month'] = energy_df['datetime'].dt.month
        energy_df['is_weekend'] = energy_df['day_of_week'].isin([5, 6]).astype(int)

        # Calculate moving averages for energy consumption
        energy_df['consumption_ma_24h'] = energy_df['energy_consumption'].rolling(window=24).mean()
        energy_df['consumption_ma_7d'] = energy_df['energy_consumption'].rolling(window=24 * 7).mean()

        # Drop rows with NaN values created by rolling averages
        energy_df = energy_df.dropna()

        return energy_df

    def preprocess_data(self, df, target_col='energy_consumption'):
        """
        Prepare data for modeling
        """
        # Select features for modeling
        feature_columns = [
            'hour', 'day_of_week', 'month', 'is_weekend',
            'temperature', 'humidity', 'wind_speed',
            'consumption_ma_24h', 'consumption_ma_7d'
        ]

        X = df[feature_columns]
        y = df[target_col]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test


# Example usage:
if __name__ == "__main__":
    # Initialize the processor
    processor = EnergyDataProcessor()

    # Load the synthetic data
    df = processor.load_energy_data('german_energy_data.csv')
    print("Data loaded successfully!")
    print("\nShape of loaded data:", df.shape)

    # Engineer features
    processed_df = processor.engineer_features(df)
    print("\nFeatures engineered successfully!")
    print("New features:", processed_df.columns.tolist())

    # Preprocess for modeling
    X_train, X_test, y_train, y_test = processor.preprocess_data(processed_df)
    print("\nData preprocessed and split successfully!")
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)



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
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
            tf.keras.layers.Dense(1)
        ])

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


import os
from data_processor import EnergyDataProcessor
from model_development import EnergyLoadModels
import joblib
import pandas as pd


def main():
    # Initialize data processor
    data_processor = EnergyDataProcessor()

    # Load synthetic energy consumption data
    print("Loading synthetic energy data...")
    energy_df = data_processor.load_energy_data('german_energy_data.csv')

    # Engineer features
    print("Engineering features...")
    processed_df = data_processor.engineer_features(energy_df)

    # Preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = data_processor.preprocess_data(processed_df)

    # Initialize model development
    print("Initializing models...")
    energy_models = EnergyLoadModels()

    # Train and evaluate models
    print("\nTraining Linear Regression baseline...")
    lr_mae, lr_rmse = energy_models.baseline_linear_regression(X_train, X_test, y_train, y_test)
    print(f"Linear Regression - MAE: {lr_mae:.2f}, RMSE: {lr_rmse:.2f}")

    print("\nTraining LSTM network...")
    lstm_mae, lstm_rmse = energy_models.lstm_network(X_train, y_train, X_test, y_test)
    print(f"LSTM - MAE: {lstm_mae:.2f}, RMSE: {lstm_rmse:.2f}")

    print("\nTraining XGBoost model...")
    xgb_mae, xgb_rmse = energy_models.xgboost_model(X_train, X_test, y_train, y_test)
    print(f"XGBoost - MAE: {xgb_mae:.2f}, RMSE: {xgb_rmse:.2f}")

    # Compare model performance
    print("\nComparing model performance...")
    performance = energy_models.compare_models()
    print("\nModel Performance Comparison:")
    print(performance)

    # Save best model (XGBoost in this case)
    print("\nSaving best model...")
    best_model = energy_models.models['xgboost']['model']
    joblib.dump(best_model, 'best_energy_load_model.pkl')

    # Save the scaler for preprocessing new data
    joblib.dump(data_processor.scaler, 'scaler.pkl')

    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()