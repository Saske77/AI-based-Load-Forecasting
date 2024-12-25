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