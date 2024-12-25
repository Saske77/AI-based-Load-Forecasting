# Energy Load Forecasting with Machine Learning

## Project Description
This project addresses a critical challenge in the energy sector: accurately predicting electricity consumption. By leveraging advanced machine learning techniques, the system forecasts energy loads with high precision, supporting grid management, renewable energy integration, and energy waste reduction.

### Key Features:
- Integration of weather data (via OpenWeather API) and time-based features.
- Implementation of multiple machine learning models: Linear Regression, ARIMA, LSTM, and XGBoost.
- Comprehensive model evaluation using metrics such as MAE, RMSE, and MAPE.
- Deployment as a Streamlit web application for real-time energy load predictions.

---

## Table of Contents
- [Installation](#installation)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [How It Works](#how-it-works)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/energy-forecasting.git
   cd energy-forecasting
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: `env\Scripts\activate`
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Obtain an API key from OpenWeather and add it to the `.env` file:
   ```env
   OPENWEATHER_API_KEY=your_api_key
   ```
5. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## Features
- **Data Collection:**
  - Historical energy consumption data.
  - Weather data using OpenWeather API.
  - Time-based features such as hour, day, month, and holidays.

- **Data Preprocessing:**
  - Handling missing values.
  - Feature engineering (temporal and weather-related).
  - Normalization and scaling.

- **Machine Learning Models:**
  - Linear Regression: Baseline model.
  - ARIMA: Traditional time-series model.
  - LSTM Neural Network: For complex pattern recognition.
  - XGBoost: Gradient boosting for structured data.

- **Evaluation Metrics:**
  - Mean Absolute Error (MAE).
  - Root Mean Squared Error (RMSE).
  - Mean Absolute Percentage Error (MAPE).

- **Deployment:**
  - Real-time predictions using Streamlit.
  - User-friendly interface for parameter input.

---

## Technologies Used
- **Programming Languages:** Python
- **Libraries:** TensorFlow, Scikit-learn, Pandas, NumPy, XGBoost, Streamlit
- **API:** OpenWeather API

---

## How It Works
1. **Data Collection:** Historical data is combined with weather and time-based features.
2. **Preprocessing:** Data is cleaned, normalized, and engineered for optimal input.
3. **Model Training:** Models are trained and evaluated on key metrics.
4. **Deployment:** The best-performing model (XGBoost) is integrated into a Streamlit app for real-time predictions.

---

## Results
- **Best Model:** XGBoost
  - MAE: 213.79
  - RMSE: 332.89

The project demonstrates significant improvements in forecasting accuracy, aiding better grid management and renewable energy integration.

---

## Future Work
- Extend the system to support multiple regions.
- Explore advanced architectures like Transformers.
- Integrate with IoT-based smart grids.
- Add real-time continuous learning capabilities.

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
