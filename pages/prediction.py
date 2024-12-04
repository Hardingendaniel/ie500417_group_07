from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
from data.data import load_total_ghg_data, load_markdown

# Path to markdown files
ARIMA_MODEL_EXPLANATION = 'data/markdown/arima_model_explanation.md'
EXPONENTIAL_SMOOTHING_EXPLANATION = 'data/markdown/exponential_smoothing_explanation.md'
ADABOOST_REGRESSION_EXPLANATION = 'data/markdown/adaboost_regression_explanation.md'
COMPARISON_GRAPH_EXPLANATION = 'data/markdown/comparison_graph_explanation.md'
MODEL_ERROR_COMPARISON = 'data/markdown/model_error_comparison.md'

# Load and preprocess data
european_countries = [
    "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium", "Bosnia and Herzegovina",
    "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Finland", "France", "Georgia", "Germany",
    "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Kazakhstan", "Kosovo", "Latvia", "Liechtenstein",
    "Lithuania", "Luxembourg", "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands", "North Macedonia",
    "Norway", "Poland", "Portugal", "Romania", "Russia", "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain",
    "Sweden", "Switzerland", "Turkey", "Ukraine", "United Kingdom", "Vatican City"
]

# Function to preprocess data
def preprocess_data(data, entities, start_year='1990', end_year='2020', resample_freq='YE'):
    data = data[data['Entity'].isin(entities)].copy()
    data['Year'] = pd.to_datetime(data['Year'], format='%Y') + pd.offsets.YearEnd(0)
    data.set_index('Year', inplace=True)
    data['total_ghg'] = pd.to_numeric(data['Annual greenhouse gas emissions in CO₂ equivalents'], errors='coerce')
    data_resampled = data['total_ghg'].resample(resample_freq).sum()
    return data_resampled.loc[f'{start_year}-12-31':f'{end_year}-12-31']

# Load and preprocess data
data = load_total_ghg_data()
annual_data = preprocess_data(data, european_countries)

# Train-Test Split
train_size = int(len(annual_data) * 0.9)
train_data, test_data = annual_data[:train_size], annual_data[train_size:]
years_to_predict = 4

# Generate future years
last_year_in_data = annual_data.index[-1].year
future_years = pd.date_range(start=f'{last_year_in_data + 1}-12-31', periods=years_to_predict, freq='YE')

# Forecast models class
class ForecastModels:
    def __init__(self, train_data):
        self.train_data = train_data

    def arima(self, order=(2, 1, 2), test_index=None):
        model = ARIMA(self.train_data, order=order, enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(method_kwargs={'maxiter': 200})
        forecast = fit.forecast(steps=len(test_index))
        return pd.Series(forecast.values, index=test_index)

    def exponential_smoothing(self, test_index=None):
        model = ExponentialSmoothing(self.train_data, trend='add', seasonal=None, initialization_method='estimated')
        fit = model.fit(optimized=True)
        forecast = fit.forecast(steps=len(test_index))
        return pd.Series(forecast.values, index=test_index)

    def adaboost(self, test_index=None, n_estimators=200, learning_rate=0.01, random_state=42):
        train_df = self.train_data.reset_index()
        train_df.columns = ['Year', 'Emissions']
        X_train = train_df['Year'].dt.year.values.reshape(-1, 1)
        y_train = train_df['Emissions'].values

        ada_regressor = AdaBoostRegressor(
            estimator=LinearRegression(),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state
        )
        ada_regressor.fit(X_train, y_train)

        future_years = test_index.year.values.reshape(-1, 1)
        predictions = ada_regressor.predict(future_years)
        return pd.Series(predictions, index=test_index)

# Initialize models and generate forecasts
forecast_models = ForecastModels(train_data)

# Forecast for validation set
forecasts = {
    'ARIMA': forecast_models.arima(test_index=test_data.index),
    'Exponential Smoothing': forecast_models.exponential_smoothing(test_index=test_data.index),
    'AdaBoost Regression': forecast_models.adaboost(test_index=test_data.index)
}

# Forecast for future years
forecasts_future = {
    'ARIMA': forecast_models.arima(test_index=future_years),
    'Exponential Smoothing': forecast_models.exponential_smoothing(test_index=future_years),
    'AdaBoost Regression': forecast_models.adaboost(test_index=future_years)
}

# Function to calculate errors for models
def calculate_errors(test, forecasts):
    errors = {}
    for model_name, forecast in forecasts.items():
        aligned_forecast = forecast[test.index]
        errors[model_name] = mean_squared_error(test, aligned_forecast)
    return errors

# Function to plot forecasts for a single model
def plot_single_forecast(actual, forecast_validation, forecast_future, model_name, title):
    fig = go.Figure()
    # Plot actual data
    fig.add_trace(go.Scatter(
        x=actual.index, y=actual.values,
        mode='lines', name='Actual Data',
        line=dict(color='blue')
    ))
    # Plot forecast on validation set
    fig.add_trace(go.Scatter(
        x=forecast_validation.index, y=forecast_validation.values,
        mode='lines', name=f'{model_name} Validation Forecast',
        line=dict(dash='dash')
    ))
    # Plot future forecast
    fig.add_trace(go.Scatter(
        x=forecast_future.index, y=forecast_future.values,
        mode='lines', name=f'{model_name} Future Forecast',
        line=dict(dash='dot')
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title='Emissions',
        hovermode='x unified'
    )
    return fig

# Generate individual forecast figures

# ARIMA
arima_validation_forecast = forecasts['ARIMA']
arima_future_forecast = forecasts_future['ARIMA']
arima_fig = plot_single_forecast(
    annual_data, arima_validation_forecast, arima_future_forecast, 'ARIMA', 'ARIMA Forecast')

# Exponential Smoothing
exp_smoothing_validation_forecast = forecasts['Exponential Smoothing']
exp_smoothing_future_forecast = forecasts_future['Exponential Smoothing']
exp_smoothing_fig = plot_single_forecast(
    annual_data, exp_smoothing_validation_forecast, exp_smoothing_future_forecast, 'Exponential Smoothing', 'Exponential Smoothing Forecast')

# AdaBoost Regression
adaboost_validation_forecast = forecasts['AdaBoost Regression']
adaboost_future_forecast = forecasts_future['AdaBoost Regression']
adaboost_fig = plot_single_forecast(
    annual_data, adaboost_validation_forecast, adaboost_future_forecast, 'AdaBoost Regression', 'AdaBoost Regression Forecast')

# Generate comparison plot including future forecasts
def plot_forecasts(actual, forecasts, forecasts_future, title):
    fig = go.Figure()
    # Plot actual data
    fig.add_trace(go.Scatter(
        x=actual.index, y=actual.values,
        mode='lines', name='Actual Data',
        line=dict(color='blue')
    ))
    # Plot forecasts on validation set
    for model_name, forecast in forecasts.items():
        fig.add_trace(go.Scatter(
            x=forecast.index, y=forecast.values,
            mode='lines', name=f'{model_name} Validation Forecast',
            line=dict(dash='dash')
        ))
    # Plot future forecasts
    for model_name, forecast in forecasts_future.items():
        fig.add_trace(go.Scatter(
            x=forecast.index, y=forecast.values,
            mode='lines', name=f'{model_name} Future Forecast',
            line=dict(dash='dot')
        ))
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title='Emissions',
        hovermode='x unified'
    )
    return fig

# Generate the comparison plot
forecast_fig = plot_forecasts(annual_data, forecasts, forecasts_future, 'Forecast Comparison Including Future Years')

# Calculate errors on validation set
errors = calculate_errors(test_data, forecasts)

# Error bar chart
error_fig = go.Figure([
    go.Bar(x=list(errors.keys()), y=list(errors.values()), name='MSE', marker_color='blue')
])
error_fig.update_layout(title='Model Error Comparison', xaxis_title='Model', yaxis_title='Mean Squared Error')

# Define Dash layout
layout = html.Div(className='centered-content', children=[
    html.H1(children='Greenhouse Gas Emissions Forecast for Europe (1990-2020)'),

    # ARIMA
    dcc.Markdown(id='arima-model-explanation', dangerously_allow_html=True, className='markdown-content'),
    dcc.Graph(id='arima-forecast-chart', figure=arima_fig),

    # Exponential Smoothing
    dcc.Markdown(id='exponential-smoothing-explanation', dangerously_allow_html=True, className='markdown-content'),
    dcc.Graph(id='exponential-smoothing-forecast-chart', figure=exp_smoothing_fig),

    # AdaBoost Regression
    dcc.Markdown(id='adaboost-regression-explanation', dangerously_allow_html=True, className='markdown-content'),
    dcc.Graph(id='adaboost-forecast-chart', figure=adaboost_fig),

    # Comparison Graph
    dcc.Markdown(id='comparison-graph-explanation', dangerously_allow_html=True, className='markdown-content'),
    dcc.Graph(id='forecast-comparison-chart', figure=forecast_fig),

    # Error Chart
    dcc.Markdown(id='model-error-comparison', dangerously_allow_html=True, className='markdown-content'),
    dcc.Graph(id='error-chart', figure=error_fig),
])

# Callbacks for updating markdown content
def init_callbacks(app):
    @app.callback(
        Output('arima-model-explanation', 'children'),
        Input('interval-component', 'n_intervals')
    )
    def update_arima_model_explanation(n_intervals):
        return load_markdown(ARIMA_MODEL_EXPLANATION)

    @app.callback(
        Output('exponential-smoothing-explanation', 'children'),
        Input('interval-component', 'n_intervals')
    )
    def update_exponential_smoothing_explanation(n_intervals):
        return load_markdown(EXPONENTIAL_SMOOTHING_EXPLANATION)

    @app.callback(
        Output('adaboost-regression-explanation', 'children'),
        Input('interval-component', 'n_intervals')
    )
    def update_adaboost_regression_explanation(n_intervals):
        return load_markdown(ADABOOST_REGRESSION_EXPLANATION)

    @app.callback(
        Output('comparison-graph-explanation', 'children'),
        Input('interval-component', 'n_intervals')
    )
    def update_comparison_graph_explanation(n_intervals):
        return load_markdown(COMPARISON_GRAPH_EXPLANATION)

    @app.callback(
        Output('model-error-comparison', 'children'),
        Input('interval-component', 'n_intervals')
    )
    def update_model_error_comparison(n_intervals):
        return load_markdown(MODEL_ERROR_COMPARISON)
