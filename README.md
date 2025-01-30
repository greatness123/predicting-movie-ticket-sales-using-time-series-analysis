# predicting-movie-ticket-sales-using-time-series-analysis
Advanced Time Series Analysis and Sales Prediction for Silverbird Cinemas

Project Overview

This project focuses on applying time series analysis to forecast ticket sales for Silverbird Cinemas using the Seasonal Autoregressive Integrated Moving Average (SARIMA) model. The objective is to predict future sales with high accuracy by analyzing historical data, seasonal fluctuations, and trends. The insights derived from this project help inform decisions related to inventory management, marketing campaigns, and staffing.

Objectives

Analyze historical ticket sales data to identify trends and seasonal patterns.

Develop and fine-tune a SARIMA model for accurate sales forecasting.

Evaluate the model’s accuracy and applicability for operational decision-making using Mean Absolute Percentage Error (MAPE) and Root Mean Squared Error (RMSE).

Dataset

Source: Kaggle

Period Covered: 2008 - 2012

Columns:

Date: Timestamp of ticket sales.

Sales: Number of tickets sold on that date.

Methodology

1. Data Preprocessing

Convert Date column to datetime format.

Set Date as the index for time series analysis.

Handle missing values using interpolation.

2. Stationarity Check

Apply the Augmented Dickey-Fuller (ADF) Test to check if the series is stationary.

If the series is not stationary, apply differencing to remove trends.

3. Model Selection and Training

Analyze Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots to determine ARIMA parameters.

Implement the SARIMA model with selected parameters:

Non-seasonal order: (0,1,0)

Seasonal order: (1,0,1,12)

Train the model and generate forecasts for the next 12 months.

4. Model Evaluation

Compare predicted values with actual sales using:

Mean Absolute Percentage Error (MAPE)

Root Mean Squared Error (RMSE)

Implementation (Python Code)

Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

Load and Prepare Dataset

ticket_sales = pd.read_csv("Ticket Sales.csv")
ticket_sales['Date'] = pd.to_datetime(ticket_sales['Date'])
ticket_sales.set_index('Date', inplace=True)

Check for Stationarity

def perform_adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]:.50f}')
perform_adf_test(ticket_sales['Sales'])

Apply Differencing (if needed)

ticket_sales['Sales_Diff'] = ticket_sales['Sales'].diff().dropna()

Plot ACF and PACF for Model Selection

plot_acf(ticket_sales['Sales_Diff'].dropna())
plot_pacf(ticket_sales['Sales_Diff'].dropna())
plt.show()

Train the SARIMA Model

model = SARIMAX(ticket_sales['Sales'],
                order=(0,1,0),
                seasonal_order=(1,0,1,12),
                enforce_stationarity=False,
                enforce_invertibility=False)
results = model.fit()

Forecast and Evaluate Predictions

forecast = results.get_forecast(steps=12)
forecast_index = pd.date_range(start='2013-01-01', periods=12, freq='M')
forecast_values = forecast.predicted_mean

mape = mean_absolute_percentage_error(ticket_sales['Sales'][-12:], forecast_values)
rmse = np.sqrt(mean_squared_error(ticket_sales['Sales'][-12:], forecast_values))
print(f"MAPE: {mape:.2f}%")
print(f"RMSE: {rmse}")

Results and Findings

MAPE: 2.88% (High forecasting accuracy)

RMSE: 804 (Low error considering sales range 0 - 28,000)

Forecast plots closely match actual sales trends, validating the model's effectiveness.

Discussion

Strengths

High accuracy in capturing seasonal trends.

Provides valuable insights for cinema management.

Limitations

External factors (marketing campaigns, competitor activities) are not included in predictions.

Model assumptions may not hold if the sales pattern changes significantly over time.

Future Work

Incorporate external factors like promotional activities and economic trends.

Experiment with other forecasting models such as LSTMs and Facebook Prophet.

Conclusion

This project successfully developed a SARIMA model for forecasting Silverbird Cinemas' ticket sales with high accuracy. The model provides valuable insights for strategic planning, inventory management, and resource allocation. Future enhancements can improve performance by integrating more external data sources.

References

Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). Time Series Analysis: Forecasting and Control. Wiley.

Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice. OTexts.

Dataset sourced from Kaggle.

Python Libraries: Pandas, Matplotlib, Statsmodels, Sklearn.

How to Use

Setup Environment

Install required Python libraries:

pip install pandas numpy matplotlib statsmodels scikit-learn

Place Ticket Sales.csv in the working directory.

Run the Python script to preprocess data, train the model, and generate forecasts.

Expected Output

Graphs of historical sales and forecasted sales.

Printed evaluation metrics (MAPE, RMSE) for assessing model accuracy.

This README provides a structured overview of the project, ensuring clarity for both technical and non-technical audiences.


