
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt


DataDir = "C:/Users/alimo/DownLoads/"
DataFile = DataDir + 'Nat_Gas.csv'


data = pd.read_csv(DataFile, parse_dates=['Dates'], index_col='Dates')
#print(data.head())

data.plot(figsize=(10, 5))
plt.title('Monthly Natural Gas Prices')
plt.ylabel('Price ($)')
plt.xlabel('Date')
plt.show()

decomposition = seasonal_decompose(data['Prices'], model='additive')
fig = decomposition.plot()
plt.show()

order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)  # Assuming yearly seasonality with monthly data
model = SARIMAX(data['Prices'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
results = model.fit(disp=False)
#print(results.summary())

forecast = results.get_forecast(steps=12)
mean_forecast = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(data.index, data['Prices'], label='observed')
plt.plot(mean_forecast.index, mean_forecast, color='r', label='forecast')
plt.fill_between(confidence_intervals.index,
                 confidence_intervals.iloc[:, 0],
                 confidence_intervals.iloc[:, 1], color='pink')
plt.title('Forecast and Confidence Intervals')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

def estimate_price(date):
    date = pd.to_datetime(date)
    if date in mean_forecast.index:
        return mean_forecast.loc[date]
    else:
        return results.get_forecast(steps=(date - data.index[-1]).days // 30).predicted_mean[-1]

# Example usage
input_date = input("Please enter a date (YYYY-MM-DD): ")

print(estimate_price(input_date))
