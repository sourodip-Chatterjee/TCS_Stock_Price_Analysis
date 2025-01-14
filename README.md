# TCS_Stock_Price_Analysis

This project analyzes TCS stock prices using Python. It provides candlestick chart visualizations and predicts future stock prices using a linear regression model.

## Features
- **Candlestick Chart Visualization:** Displays stock price trends over time.
- **Stock Price Prediction:** Predicts future stock prices for the next 5 business days.
- **Data Preprocessing:** Handles date extraction and feature engineering.
- **Interactive Plots:** Uses Plotly for dynamic and user-friendly charts.

## Requirements
To run this project, you need Python 3.8 or above and the following libraries:

- `pandas`
- `matplotlib`
- `scikit-learn`
- `plotly`

You can install the required dependencies by running:
```bash
pip install -r requirements.txt
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/TCS_Stock_Price_Analysis.git
   ```

2. Navigate to the project directory:
   ```bash
   cd TCS_Stock_Price_Analysis
   ```

3. Place the `TCS.csv` file in the `data/` folder.

4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the script:
   ```bash
   python tcs_stock_analysis.py
   ```

2. The script performs the following tasks:
   - Validates and preprocesses the input dataset.
   - Displays a candlestick chart for TCS stock price analysis.
   - Trains a linear regression model on historical data.
   - Predicts stock prices for the next 5 business days.
   - Plots the actual vs. predicted stock prices.

3. Check the console for the predicted stock prices and enjoy the interactive chart visualizations.

## Project Structure
```
TCS_Stock_Price_Analysis/
├── tcs_stock_analysis.py      # Main Python script
├── README.md                  # Project description and usage
├── requirements.txt           # List of dependencies
├── data/                      # Folder for storing the TCS.csv file
│   └── TCS.csv                # Sample dataset (optional)
└── .gitignore                 # Files and folders to ignore in the repository
```

## Sample Output
### Predicted Stock Prices
The script generates predicted stock prices for the next 5 business days and displays them in the console, along with a visualization comparing actual and predicted prices.

### Candlestick Chart
An interactive candlestick chart is displayed to visualize stock price trends.

## Future Improvements
- Integrate advanced machine learning models for better predictions.
- Fetch real-time stock data using APIs.
- Add a user interface for enhanced interactivity.


---

For any issues or suggestions, please open an issue on the [GitHub repository](https://github.com/sourodip-Chatterjee/TCS_Stock_Price_Analysis).

