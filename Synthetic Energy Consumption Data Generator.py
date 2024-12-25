import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_german_energy_data(start_date='2023-01-01', days=365):
    """
    Generate synthetic energy consumption data based on German energy patterns.

    Parameters:
    - start_date: Starting date for the dataset
    - days: Number of days to generate data for

    German-specific characteristics:
    - Peak consumption: Winter months (December-February)
    - Lower consumption: Summer months (June-August)
    - Industrial working hours: Typically 6:00-18:00
    - Public holidays: German calendar
    - Renewable energy influence: Higher solar in summer
    """
    # Create date range with hourly frequency
    date_rng = pd.date_range(start=start_date, periods=days * 24, freq='H')

    # German public holidays (simplified list)
    german_holidays = [
        '2023-01-01',  # New Year
        '2023-04-07',  # Good Friday
        '2023-04-10',  # Easter Monday
        '2023-05-01',  # Labor Day
        '2023-05-18',  # Ascension Day
        '2023-05-29',  # Whit Monday
        '2023-10-03',  # German Unity Day
        '2023-12-25',  # Christmas Day
        '2023-12-26'  # Boxing Day
    ]

    # Base load for German industrial area
    base_load = 2500  # kWh (higher due to industrial base)

    data = []
    for dt in date_rng:
        # Hour of day effect (German industrial working hours)
        if 6 <= dt.hour <= 18:
            hour_effect = np.sin((dt.hour - 6) * np.pi / 12) * 1000 + 1000
        else:
            hour_effect = 500  # Lower night consumption

        # Day of week effect
        if dt.weekday() >= 5:  # Weekend
            weekend_effect = 0.6  # German industry typically closed on weekends
        else:
            weekend_effect = 1.0

        # Seasonal effect (stronger in Germany due to heating/cooling needs)
        # More pronounced winter peak, less summer variation
        month = dt.month
        if month in [12, 1, 2]:  # Winter
            season_effect = 1500
        elif month in [6, 7, 8]:  # Summer
            season_effect = 800
        else:  # Spring/Autumn
            season_effect = 1000

        # Holiday effect
        holiday_effect = 0.5 if dt.strftime('%Y-%m-%d') in german_holidays else 1.0

        # Temperature effect (German climate)
        temp_base = 20 - (np.sin(dt.month * np.pi / 6) * 15)  # Temperature range -5 to 25°C
        temperature = temp_base + np.random.normal(0, 2)

        # Energy consumption calculation
        consumption = (base_load + hour_effect + season_effect) * weekend_effect * holiday_effect

        # Add temperature dependency
        if temperature < 15:  # Heating effect
            consumption += (15 - temperature) * 50
        elif temperature > 22:  # Cooling effect (less pronounced in Germany)
            consumption += (temperature - 22) * 30

        # Random variation (smaller for more stability)
        noise = np.random.normal(0, consumption * 0.05)
        consumption += noise

        # Ensure no negative values
        consumption = max(consumption, 0)

        data.append({
            'datetime': dt,
            'energy_consumption': round(consumption, 2),
            'temperature': round(temperature, 2),
            'humidity': round(min(max(np.random.normal(70, 10), 0), 100), 2),  # German humidity patterns
            'wind_speed': round(max(np.random.normal(6, 2), 0), 2),  # German wind patterns
            'is_holiday': 1 if dt.strftime('%Y-%m-%d') in german_holidays else 0
        })

    df = pd.DataFrame(data)
    return df


# Generate sample data
df = generate_german_energy_data()

# Add realistic anomalies (factory shutdowns, grid maintenance)
random_indices = np.random.choice(len(df), size=int(len(df) * 0.005), replace=False)
df.loc[random_indices, 'energy_consumption'] *= np.random.uniform(0.4, 0.6)

# Save to CSV
df.to_csv('german_energy_data.csv', index=False)

# Print sample statistics
print("\nGerman Energy Dataset Statistics:")
print(f"Total Records: {len(df)}")
print("\nEnergy Consumption Statistics (kWh):")
print(df['energy_consumption'].describe())
print("\nTemperature Statistics (°C):")
print(df['temperature'].describe())
print("\nSample of the data:")
print(df.head())