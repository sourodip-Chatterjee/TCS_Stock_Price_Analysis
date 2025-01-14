import os
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Constants
REQUIRED_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close']
FILE_PATH = "C:\\Users\\Admin\\Desktop\\TCS.csv"

# Functions
def load_data(filepath):
    """Load stock data from a CSV file."""
    try:
        data = pd.read_csv(filepath)
        return data
    except FileNotFoundError:
        print("Error: File not found. Please check the file path.")
        exit()

def validate_data(data):
    """Validate if the dataset contains all required columns."""
    if not all(col in data.columns for col in REQUIRED_COLUMNS):
        raise ValueError(f"Dataset is missing one or more required columns: {REQUIRED_COLUMNS}")

def preprocess_data(data):
    """Preprocess the data by adding day, month, and year columns."""
    data['Date'] = pd.to_datetime(data['Date'])
    data['day'] = data['Date'].dt.day
    data['month'] = data['Date'].dt.month
    data['year'] = data['Date'].dt.year
    return data

def visualize_data(data):
    """Create and display a candlestick chart."""
    figure = go.Figure(data=[
        go.Candlestick(
            x=data["Date"],
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"]
        )
    ])
    figure.update_layout(
        title="TCS Stock Price Analysis",
        xaxis_rangeslider_visible=False
    )
    figure.show()

def train_model(data, features, target):
    """Train a linear regression model."""
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict_future(data, model, features):
    """Predict future stock prices."""
    future_dates = pd.date_range(data['Date'].max() + pd.DateOffset(1), periods=5, freq='B')
    future_data = pd.DataFrame({'Date': future_dates})
    future_data['day'] = future_data['Date'].dt.day
    future_data['month'] = future_data['Date'].dt.month
    future_data['year'] = future_data['Date'].dt.year

    # Use the last known values for Open, High, and Low
    future_data['Open'] = data['Open'].iloc[-1]
    future_data['High'] = data['High'].iloc[-1]
    future_data['Low'] = data['Low'].iloc[-1]

    # Predict Close prices
    future_data['Predicted_Close'] = model.predict(future_data[features])

    # Adjust predictions using historical standard deviation
    historical_std = data['Close'].pct_change().std()
    future_data['Predicted_Close'] *= (1 + historical_std)

    return future_data

def plot_predictions(data, future_data):
    """Plot actual and predicted stock prices."""
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], label='Actual Close Price')
    plt.plot(future_data['Date'], future_data['Predicted_Close'], 
             label='Predicted Close Price', linestyle='dashed', marker='o')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

# Main Script
def main():
    # Step 1: Load Data
    data = load_data(FILE_PATH)

    # Step 2: Validate Data
    validate_data(data)

    # Step 3: Preprocess Data
    data = preprocess_data(data)

    # Step 4: Visualize Data
    visualize_data(data)

    # Step 5: Train Model
    features = ['day', 'month', 'year', 'Open', 'High', 'Low']
    target = 'Close'
    model = train_model(data, features, target)

    # Step 6: Predict Future Stock Prices
    future_data = predict_future(data, model, features)

    # Step 7: Display Predictions
    print("\nPredicted Stock Prices for the Next 5 Days:")
    print(future_data[['Date', 'Predicted_Close']])

    # Step 8: Plot Predictions
    plot_predictions(data, future_data)

if __name__ == "__main__":
    main()