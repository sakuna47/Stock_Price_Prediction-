# Stock Price Prediction Web App

## Overview
This project is a **Stock Price Prediction Web App** that uses **Machine Learning (Random Forest Regressor)** to predict the next day's closing stock price based on historical stock data. The application is built using **Python, Scikit-learn, Pandas, Seaborn, Matplotlib, and Streamlit**.

## Features
- Fetches historical stock data using **Yahoo Finance (yfinance)**.
- Performs data preprocessing, including handling missing values and detecting outliers.
- Uses a **Random Forest Regressor** model to predict stock prices.
- Provides a **Streamlit web app** for easy user interaction.
- Saves and loads trained models using **pickle**.

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- pip

### Install Required Libraries
```bash
pip install pandas numpy scikit-learn yfinance imbalanced-learn matplotlib seaborn streamlit
```

## Usage

### Running the Model
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock-price-prediction.git
   cd stock-price-prediction
   ```

2. Run the Python script to train and save the model:
   ```bash
   python stock_prediction.py
   ```

3. Start the Streamlit web app:
   ```bash
   streamlit run app.py
   ```

### Using the Web App
1. Enter a **stock ticker symbol** (e.g., `AAPL` for Apple Inc.).
2. Click the **Predict** button to fetch stock data and predict the next closing price.
3. View the predicted closing price for the next trading day.

## Project Structure
```
📂 stock-price-prediction
│── 📜 stock_prediction.py   # Training & model-saving script
│── 📜 app.py                # Streamlit web app script
│── 📜 stock_model.pkl       # Saved Random Forest model
│── 📜 scaler.pkl            # Saved StandardScaler object
│── 📜 README.md             # Project documentation
```

## Model Training Process
- Downloads **historical stock data** using `yfinance`.
- Handles missing values using forward fill (`ffill`).
- Removes outliers using **Interquartile Range (IQR)**.
- Splits data into **train and test sets** (80-20 split).
- Scales features using **StandardScaler**.
- Trains a **Random Forest Regressor** model.
- Evaluates the model using **Mean Squared Error (MSE)** and **R2 Score**.
- Saves the trained model and scaler using `pickle`.

## Technologies Used
- **Python**
- **Scikit-learn** (Machine Learning Model)
- **Pandas & NumPy** (Data Manipulation)
- **Matplotlib & Seaborn** (Data Visualization)
- **Yahoo Finance API** (Stock Data Fetching)
- **Streamlit** (Web App Interface)
- **Google Colab** (Training Environment)

## Future Improvements
- Integrate **Deep Learning models (LSTMs, GRUs)** for better predictions.
- Implement real-time stock data fetching and predictions.
- Add **multiple stock predictions** and trend analysis.

## License
This project is licensed under the MIT License.

## Author
[sakuna sankalpa](https://github.com/sakuna47)

