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


import matplotlib.pyplot as plt

# After training models and collecting metrics
def plot_model_comparison(performance):
    """
    Plots a bar graph to compare models' performance metrics.

    Parameters:
        performance (pd.DataFrame): A DataFrame containing model names, MAE, and RMSE.
    """
    # Extract data for plotting
    model_names = performance['Model']
    mae_values = performance['MAE']
    rmse_values = performance['RMSE']

    # Bar graph setup
    x = range(len(model_names))  # Positions on x-axis
    width = 0.4  # Bar width

    plt.figure(figsize=(10, 6))
    # Plot MAE
    plt.bar([pos - width / 2 for pos in x], mae_values, width, label='MAE')
    # Plot RMSE
    plt.bar([pos + width / 2 for pos in x], rmse_values, width, label='RMSE')

    # Customize the chart
    plt.title('Model Performance Comparison', fontsize=16)
    plt.xlabel('Models', fontsize=14)
    plt.ylabel('Error Metrics', fontsize=14)
    plt.xticks(x, model_names, fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()

energy_models = EnergyLoadModels()  # Ensure the object is initialized
performance = energy_models.compare_models()
print("\nModel Performance Comparison:")
print(performance)





