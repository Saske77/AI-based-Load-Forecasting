import streamlit as st
import numpy as np
import pandas as pd
import joblib


def load_model():
    """Load the best performing model"""
    return joblib.load('best_energy_load_model.pkl')


def predict_load(model, input_features):
    """Make predictions using the loaded model"""
    prediction = model.predict(input_features)
    return prediction[0]


def main():
    st.title('Energy Load Forecasting')

    # Sidebar for input features
    st.sidebar.header('Input Energy Load Parameters')

    hour = st.sidebar.slider('Hour of Day', 0, 23, 12)
    day_of_week = st.sidebar.selectbox('Day of Week',
                                       ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                                        'Friday', 'Saturday', 'Sunday'])
    month = st.sidebar.selectbox('Month',
                                 ['January', 'February', 'March', 'April', 'May',
                                  'June', 'July', 'August', 'September', 'October',
                                  'November', 'December'])
    temperature = st.sidebar.number_input('Temperature (°C)', min_value=-10.0, max_value=50.0, value=20.0)
    humidity = st.sidebar.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=50.0)
    wind_speed = st.sidebar.number_input('Wind Speed (m/s)', min_value=0.0, max_value=50.0, value=5.0)

    # Add inputs for moving averages
    consumption_ma_24h = st.sidebar.number_input('24-hour Moving Average Consumption (kWh)', min_value=0.0, value=100.0)
    consumption_ma_7d = st.sidebar.number_input('7-day Moving Average Consumption (kWh)', min_value=0.0, value=150.0)

    # Convert categorical inputs
    day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                   'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5,
                     'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10,
                     'November': 11, 'December': 12}

    input_data = np.array([[
        hour,
        day_mapping[day_of_week],
        month_mapping[month],
        1 if day_of_week in ['Saturday', 'Sunday'] else 0,
        temperature,
        humidity,
        wind_speed,
        consumption_ma_24h,
        consumption_ma_7d
    ]])

    model = load_model()

    if st.sidebar.button('Predict Energy Load'):
        prediction = predict_load(model, input_data)
        st.success(f'Predicted Energy Load: {prediction:.2f} kWh')

        # Optional: Display input parameters
        st.write('Input Parameters:')
        st.write(pd.DataFrame(input_data,
                              columns=['hour', 'day_of_week', 'month', 'is_weekend', 'temperature', 'humidity',
                                       'wind_speed', 'consumption_ma_24h', 'consumption_ma_7d']))


if __name__ == '__main__':
    main()
