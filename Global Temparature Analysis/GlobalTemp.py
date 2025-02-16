import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys

def analyze_region(region_name, df):
    # Filter data for the selected region
    region_df = df[df['Country'] == region_name].copy()
    
    # Check if data exists
    if len(region_df) == 0:
        print(f"\nNo data available for {region_name}. Exiting program.")
        sys.exit()
    
    # Data preprocessing
    region_df['dt'] = pd.to_datetime(region_df['dt'])
    region_df['Year'] = region_df['dt'].dt.year
    region_df['Month'] = region_df['dt'].dt.month

    # Data validation
    print(f"\nData points for {region_name}: {len(region_df)}")
    print(f"Year range: {region_df['Year'].min()} to {region_df['Year'].max()}")
    missing_temp = region_df['AverageTemperature'].isna().sum()
    print(f"Missing temperature values: {missing_temp} ({missing_temp/len(region_df):.1%})")

    # Filter data from 1850 onwards
    region_df = region_df[region_df['Year'] >= 1850]
    if len(region_df) == 0:
        print(f"No data available from 1850 onwards. Exiting.")
        sys.exit()
    
    missing_temp = region_df['AverageTemperature'].isna().sum()
    if missing_temp/len(region_df) > 0.3:
        print("WARNING: Over 30% data missing. Predictions may be unreliable.")
    
    region_df = region_df.dropna(subset=['AverageTemperature'])
    print(f"Final data points: {len(region_df)}")

    # Yearly averages
    yearly_temp = region_df.groupby('Year')['AverageTemperature'].mean()
    
    # Plot yearly temperature
    plt.figure(figsize=(15, 7))
    plt.plot(yearly_temp.index, yearly_temp.values)
    plt.title(f'Yearly Average Temperature in {region_name} (1850-2013)')
    plt.xlabel('Year')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    plt.show()

    # Machine learning preparation
    X = yearly_temp.index.values.reshape(-1, 1)
    y = yearly_temp.values
    
    if len(X) < 10:
        print("Insufficient data for modeling.")
        return
    
    # Train-test split and scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    print(f"\nLinear Regression - R²: {r2_score(y_test, lr_pred):.2f}")

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    print(f"Random Forest - R²: {r2_score(y_test, rf_pred):.2f}")

    # Future predictions
    future_years = np.arange(2014, 2051).reshape(-1, 1)
    future_scaled = scaler.transform(future_years)
    
    plt.figure(figsize=(15, 7))
    plt.plot(yearly_temp.index, yearly_temp.values, label='Historical')
    plt.plot(future_years, lr_model.predict(future_scaled), 'r--', label='Linear Regression Forecast')
    plt.plot(future_years, rf_model.predict(future_scaled), 'g--', label='Random Forest Forecast')
    plt.title(f'Temperature Projections for {region_name} (2014-2050)')
    plt.xlabel('Year')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.show()

def analyze_global(df):
    # Global processing
    global_df = df.copy()
    global_df['dt'] = pd.to_datetime(global_df['dt'])
    global_df['Year'] = global_df['dt'].dt.year
    global_df = global_df[global_df['Year'] >= 1850].dropna(subset=['AverageTemperature'])
    
    yearly_global = global_df.groupby('Year')['AverageTemperature'].mean()
    
    # Plot global temperatures
    plt.figure(figsize=(15, 7))
    plt.plot(yearly_global.index, yearly_global.values)
    plt.title('Global Average Temperature (1850-2013)')
    plt.xlabel('Year')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    plt.show()

    # Machine learning modeling
    X = yearly_global.index.values.reshape(-1, 1)
    y = yearly_global.values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models and predictions
    lr_model = LinearRegression().fit(X_train_scaled, y_train)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_scaled, y_train)
    
    # Future projections
    future_years = np.arange(2014, 2051).reshape(-1, 1)
    future_scaled = scaler.transform(future_years)
    
    plt.figure(figsize=(15, 7))
    plt.plot(yearly_global.index, yearly_global.values, label='Historical')
    plt.plot(future_years, lr_model.predict(future_scaled), 'r--', label='Linear Regression Forecast')
    plt.plot(future_years, rf_model.predict(future_scaled), 'g--', label='Random Forest Forecast')
    plt.title('Global Temperature Projections (2014-2050)')
    plt.xlabel('Year')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load data
    print("Loading dataset...")
    try:
        df = pd.read_csv('GlobalLandTemperaturesByCountry.csv')
    except FileNotFoundError:
        print("Dataset file not found!")
        sys.exit()
    
    # User input
    valid_countries = df['Country'].unique()
    print("\nAvailable regions:", sorted(valid_countries))
    
    while True:
        user_choice = input("\nEnter a country/continent or 'Global' (q to quit): ").strip()
        if user_choice.lower() == 'q':
            break
        if user_choice == 'Global':
            analyze_global(df)
        elif user_choice in valid_countries:
            analyze_region(user_choice, df)
        else:
            print("Invalid input! Please choose from available regions or 'Global'.")
    
    print("Program exited.")