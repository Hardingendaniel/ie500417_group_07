# Model Error Comparison

This bar chart compares the Mean Squared Error (MSE) of the three forecasting models used to predict greenhouse gas (GHG)
emissions in Europe. MSE quantifies the average squared difference between the predicted values and the actual data, with 
lower values indicating better model performance.

### Models Evaluated:
- ARIMA:
  - Lowest MSE, indicating it captures the data trends most effectively and provides the most accurate forecasts among the models evaluated.
- Exponential Smoothing:
  - Slightly higher MSE than ARIMA, suggesting it has strong predictive accuracy but is marginally less optimal for this dataset.
- AdaBoost Regression:
  - Highest MSE among the three models, indicating it struggles to fit the trends in the data as effectively as ARIMA and Exponential Smoothing.

### Insights:
- **ARIMA** Is the most accurate model of the three. Arima is powerful for time series forecasting, and it performs well in this context.
- **Exponential Smoothing** Performs well, with the second lowest MSE value. With slightly higher MSE values than the ARIMA.
It effectively captures data trends and provides reliable forecasts.
- **AdaBoost Regression** shows higher error levels, This model needed some tuning which made it better, 
but it still has the highest MSE value among the three models. indicating that it may not be the best choice 
for this specific forecasting task without further optimization.

### Notes:
- Error values are displayed on the y-axis in MSE units, providing a quantitative measure of each model's forecasting accuracy.
- The reason the MSE values are in such high values are because the GHG emissions data is on a large scale, 
and the errors are squared, leading to larger values.
