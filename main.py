# from os import system
# system("pip install yfinance scikit-learn numpy matplotlib")
# system("clear || cls")
import tkinter as tkinter
from tkinter import ttk
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from numpy import array
from matplotlib.pyplot import subplots
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class StockGUI:
    def __init__(self, master):
        self.master = master
        master.title("Stock Price Prediction using AI/ML")
        self.stock_label = ttk.Label(master, text="Enter Stock Ticker:")
        self.stock_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.stock_entry = ttk.Entry(master)
        self.stock_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.predict_button = ttk.Button(master, text="Predict Price", command=self.predict_stock)
        self.predict_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
        self.prediction_label = ttk.Label(master, text="Predicted Closing Price:")
        self.prediction_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.result_label = ttk.Label(master, text="")
        self.result_label.grid(row=2, column=1, padx=10, pady=10, sticky="ew")
        self.plot_frame = ttk.LabelFrame(master, text="Historical Price with Prediction")
        self.plot_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        self.fig, self.ax = subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tkinter.BOTH, expand=True)
        master.grid_columnconfigure(1, weight=1)
        master.grid_rowconfigure(3, weight=1)
        self.model = None
        self.scaler = None
        self.historical_data = None

    def predict_stock(self):
        ticker = self.stock_entry.get().upper()
        if not ticker:
            self.result_label.config(text="Please enter a stock ticker.")
            return
        try:
            data = yf.download(ticker, period="1y")
            if data.empty:
                self.result_label.config(text=f"Could not retrieve data for {ticker}.")
                return
            self.historical_data = data['Close'].values.reshape(-1, 1)
            self.scaler = MinMaxScaler()
            scaled_data = self.scaler.fit_transform(self.historical_data)
            look_back = 60
            X, y =[],[]
            if len(scaled_data) > look_back:
                for i in range(look_back, len(scaled_data)):
                    X.append(scaled_data[i - look_back:i, 0])
                    y.append(scaled_data[i, 0])
                X = array(X)
                y = array(y)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                self.model = LinearRegression()
                self.model.fit(X_train, y_train)
                last_sequence = scaled_data[-look_back:]
                last_sequence_reshaped = last_sequence.reshape(1, look_back)
                predicted_scaled = self.model.predict(last_sequence_reshaped)
                predicted_price = self.scaler.inverse_transform(predicted_scaled.reshape(-1, 1))
                self.result_label.config(text=f"${predicted_price[0][0]:.2f}")
                self.plot_prediction(data.index[-len(y_test):], y_test, predicted_scaled)

            else:
                self.result_label.config(text="Not enough data to make a prediction.")
                self.ax.clear()
                self.ax.set_title("Historical Price")
                self.ax.plot(data['Close'].index, data['Close'].values)
                self.canvas.draw()


        except Exception as e:
            self.result_label.config(text=f"Error: {e}")

    def plot_prediction(self, dates, actual_scaled, predicted_scaled):
        if self.historical_data is not None:
            self.ax.clear()
            self.ax.set_title("Historical Price with Prediction")
            all_dates = yf.download(self.stock_entry.get().upper(), period="1y").index
            self.ax.plot(all_dates, self.historical_data, label='Historical Price', color='blue')
            actual_prices = self.scaler.inverse_transform(array(actual_scaled).reshape(-1, 1))
            predicted_prices = self.scaler.inverse_transform(array(predicted_scaled).reshape(-1, 1))
            self.ax.plot(dates, actual_prices, label='Actual Price', color='green')
            next_day = dates[-1] + (dates[-1] - dates[-2])
            self.ax.plot(next_day, predicted_prices[0][0], 'ro', label='Predicted Next Price')
            self.ax.set_xlabel("Date")
            self.ax.set_ylabel("Stock Price")
            self.ax.legend()
            self.fig.autofmt_xdate()
            self.canvas.draw()

if __name__ == "__main__":
    root = tkinter.Tk()
    app = StockGUI(root)
    root.mainloop()
