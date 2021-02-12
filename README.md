# Reliance-Stock-Market-Data-Analysis
## Time Series Analysis and Forecasting of the Reliance Stock Market Data 

### Introduction
Stock market prediction aims to determine the future movement of the stock value of a financial exchange. This project explores the Reliance Stock Market Data. Reliance Industries Limited (RIL) is an Indian multinational conglomerate company headquartered in India. The objective of this project is to forecast the ‘Close’ prices of a stock based using different modeling methods.

### Data
The dataset is the Reliance Stock Market Data of the Nifty-50 index and consists of the price history and trading volumes from the National Stock Exchange (NSE) in India. The time series spans from January 1, 2000 to July 31, 2020 and is collected daily. The dataset has many important features such as the ‘Close’, ‘Open’, ‘High’, ‘Low’, ‘Date’, ‘Volume’, and so on. These features consider the pricing history of a stock and the trading volumes.The data is taken from Kaggle and can be found here: https://www.kaggle.com/rohanrao/nifty50-stock-market-data

### EDA
EDA comprises ACF plot, Heatmap and Time Series Decomposition.

### Modeling
 - Base Models
 - Holt-Winter’s Method
 - Multiple Linear Regression
 - ARMA Model
 - ARIMA Model

### Final Model
To select the final best model the Q values and MSE of the prediction were considered. The ARIMA(2,1,2) model is the best performing one. This model has the least Q value (0.019) and MSE for prediction (1.055e+03) compared to all models. 

### Summary
The project covers a wide range of modeling and time series analysis techniques and concepts, however, for stock market given its unpredictability, it would be better to used more advanced models including neural networks such as the LSTM models. Prophet and Auto-ARIMA models are also used commonly today.
