# 📈 Stock Price Predictor using AI/ML

This is a Python-based desktop application built with Tkinter that predicts the next day's stock closing price using machine learning (Linear Regression). It uses historical stock data from Yahoo Finance and visually displays the predicted and historical prices on a plot.

## 🧠 Features

- Simple and clean GUI built with Tkinter.
- Fetches real-time stock data using [yFinance](https://pypi.org/project/yfinance/).
- Uses `LinearRegression` from Scikit-learn for prediction.
- Preprocessing with `MinMaxScaler`.
- Interactive graph powered by Matplotlib embedded within the GUI.
- Predicts next-day stock closing price based on the last 60 days.

## 🛠️ Requirements

To run this app, you need to install the following Python packages:

```bash
pip install yfinance scikit-learn numpy matplotlib
git clone https://github.com/yourusername/stock-price-predictor.git
cd stock-price-predictor
python main.py

```
