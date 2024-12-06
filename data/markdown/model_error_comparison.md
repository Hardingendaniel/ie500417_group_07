# Model Error Comparison

This bar chart compares the Mean Squared Error (MSE) of the three forecasting models used to predict greenhouse gas (GHG)
emissions in Europe. MSE quantifies the average squared difference between the predicted values and the actual data, with 
lower values indicating better model performance.

### Models Evaluated:
- Exponential Smoothing:
  - Lowest MSE, indicating it captures the data trends most effectively and provides the most accurate 
forecasts among the models evaluated.
- ARIMA:
  - Performs well with a slightly higher MSE than Exponential Smoothing, suggesting strong predictive accuracy but slightly less optimal for this dataset.
- AdaBoost Regression:
  - Exhibits the highest MSE among the three models, suggesting it may struggle to fit the trends in the data as effectively as the other methods.
  - 
### Insights:
- **Exponential Smoothing** is the most accurate model for this dataset, effectively modeling the underlying trends in greenhouse gas emission
- **ARIMA** also demonstrates good predictive capability, but its higher MSE compared to Exponential Smoothing indicates it may not capture all nuances in the data.
- **AdaBoost Regression** shows higher error levels, highlighting limitations in its applicability for this specific time series data.

### Notes:
- Error values are displayed on the y-axis in MSE units, providing a quantitative measure of each model's forecasting accuracy.
- The reason the MSE values are in such high values are because the GHG emissions data is on a large scale, 
and the errors are squared, leading to larger values.
