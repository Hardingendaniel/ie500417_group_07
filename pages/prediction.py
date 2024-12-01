from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np
import plotly.express as px
from sklearn.ensemble import AdaBoostRegressor
from data.data import data

# List of European countries
european_countries = [
    "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium", "Bosnia and Herzegovina",
    "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Finland", "France", "Georgia", "Germany",
    "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Kazakhstan", "Kosovo", "Latvia", "Liechtenstein",
    "Lithuania", "Luxembourg", "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands", "North Macedonia",
    "Norway", "Poland", "Portugal", "Romania", "Russia", "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain",
    "Sweden", "Switzerland", "Turkey", "Ukraine", "United Kingdom", "Vatican City"
]

# Filter Data for European countries
europe_data = data[data['country'].isin(european_countries)].dropna(subset=["total_ghg", "year"]).copy()
europe_data['year'] = europe_data['year'].astype(int)
europe_data['Year'] = pd.to_datetime(europe_data['year'], format='%Y') + pd.offsets.YearEnd(0)
europe_data.set_index('Year', inplace=True)
europe_data['total_ghg'] = pd.to_numeric(europe_data['total_ghg'], errors='coerce')
europe_data = europe_data.dropna(subset=['total_ghg'])

# Resample Data to Annual Data
annual_data = europe_data['total_ghg'].resample('YE').sum()

# Train-Test Split
train_size = int(len(annual_data) * 0.9)
train, test = annual_data[:train_size], annual_data[train_size:]
years_to_predict = 4

# ARIMA Model Forecast
def forecast_arima(train, test, years_to_predict, p=2, d=1, q=2):
        model_arima = ARIMA(train, order=(p, d, q))
        model_arima_fit = model_arima.fit(method_kwargs={'maxiter': 200})
        forecast_arima = model_arima_fit.forecast(steps=years_to_predict)
        last_year_arima = annual_data.index[-1].year
        forecast_index_arima = pd.date_range(
            start=f'{last_year_arima}-12-31', periods=len(test), freq='YE'
        )
        return forecast_arima, forecast_index_arima

# Exponential Smoothing
def forecast_es(annual_data, years_to_predict):
    model_es = ExponentialSmoothing(
        annual_data,
        trend='add',
        seasonal=None,
        initialization_method='legacy-heuristic'
    )
    model_fit = model_es.fit()
    last_year = annual_data.index[-1].year
    forecast_index_es = pd.date_range(
        start=f'{last_year}',
        periods=years_to_predict,
        freq='YE'
    )
    forecast_es = model_fit.forecast(steps=years_to_predict)

    return forecast_es, forecast_index_es

# Polynomial Regression
def polynomial_regression_forecast(train, test, annual_data, degree_range=[1, 2]):
    train_df = train.reset_index()
    X_train = train_df['Year'].dt.year.values.reshape(-1, 1)
    y_train = train_df['total_ghg'].values

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('linear', LinearRegression())
    ])

    param_grid = {
        'poly__degree': degree_range
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_degree = grid_search.best_params_['poly__degree']
    best_model = grid_search.best_estimator_

    forecast_steps = len(test)
    last_year = annual_data.index[-1].year
    future_years = np.arange(last_year, last_year + forecast_steps).reshape(-1, 1)
    predictions = best_model.predict(future_years)

    future_predictions_poly = pd.DataFrame({
        'Year': future_years.flatten(),
        'Predicted_total_ghg': predictions
    })
    future_predictions_poly['Year'] = pd.to_datetime(future_predictions_poly['Year'], format='%Y') + pd.offsets.YearEnd(0)

    return future_predictions_poly

# AdaBoost Regression
def adaboost_regression_forecast(train,
                                 test,
                                 annual_data,
                                 forecast_steps,
                                 n_estimators=200,
                                 learning_rate=0.01,
                                 random_state=42):

    train_df = train.reset_index()
    X_train = train_df['Year'].dt.year.values.reshape(-1, 1)
    y_train = train_df['total_ghg'].values

    ada_regressor = AdaBoostRegressor(
        estimator=LinearRegression(),
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )

    ada_regressor.fit(X_train, y_train)

    X_test = test.index.year.values.reshape(-1, 1)
    predictions_ada = ada_regressor.predict(X_test)

    last_year = annual_data.index[-1].year
    future_years_ada = np.arange(last_year + 1, last_year + 1 + forecast_steps).reshape(-1, 1)
    predictions_ada_future = ada_regressor.predict(future_years_ada)

    future_predictions_ada = pd.DataFrame({
        'Year': future_years_ada.flatten(),
        'Predicted_total_ghg': predictions_ada_future
    })

    return future_predictions_ada

# Create the ARIMA forecast chart
forecast_arima, forecast_index_arima = forecast_arima(train, test, years_to_predict)
fig_arima = go.Figure()

fig_arima.add_trace(go.Scatter(
    x=annual_data.index,
    y=annual_data,
    mode='lines',
    name='Actual Data',
    line=dict(color='red')
))

fig_arima.add_trace(go.Scatter(
    x=forecast_index_arima,
    y=forecast_arima,
    mode='lines',
    name='ARIMA Forecast',
    line=dict(color='blue', dash='dash')
))

fig_arima.update_layout(
    title='Total Greenhouse Gas Emissions Over Time with ARIMA Forecast',
    xaxis_title='Year',
    yaxis_title='Total Greenhouse Gas Emissions',
    legend=dict(x=0, y=1),
    hovermode='x unified'
)

# Exponential Smoothing Forecast chart
forecast_es, forecast_index_es = forecast_es(annual_data, years_to_predict)
fig_es = px.line(
    x=annual_data.index,
    y=annual_data,
    title="Total Greenhouse Gas Emissions Over Time with Exponential Smoothing Forecast",
    labels={'total_ghg': 'Emissions', 'Year': 'Year'}
)

fig_es.add_scatter(x=forecast_index_es, y=forecast_es, mode='lines', name='Forecast')

# Polynomial Regression Forecast chart
future_predictions_poly = polynomial_regression_forecast(train, test, annual_data)
fig_poly = go.Figure()

fig_poly.add_trace(go.Scatter(
    x=annual_data.index,
    y=annual_data,
    mode='lines',
    name='Actual Data',
    line=dict(color='red')
))

fig_poly.add_trace(go.Scatter(
    x=future_predictions_poly['Year'],
    y=future_predictions_poly['Predicted_total_ghg'],
    mode='lines',
    name='Polynomial Regression Forecast',
    line=dict(color='green', dash='longdash')
))

fig_poly.update_layout(
    title='Total Greenhouse Gas Emissions Over Time with Polynomial Regression Forecast',
    xaxis_title='Year',
    yaxis_title='Total Greenhouse Gas Emissions',
    legend=dict(x=0, y=1),
    hovermode='x unified'
)

# AdaBoost Forecast chart
future_predictions_ada = adaboost_regression_forecast(train, test, annual_data, years_to_predict)
fig_ada = go.Figure()

fig_ada.add_trace(go.Scatter(
    x=annual_data.index,
    y=annual_data,
    mode='lines',
    name='Actual Data',
    line=dict(color='red')
))

fig_ada.add_trace(go.Scatter(
    x=future_predictions_ada['Year'],
    y=future_predictions_ada['Predicted_total_ghg'],
    mode='lines',
    name='AdaBoost Forecast',
    line=dict(color='purple', dash='dot')
))

fig_ada.update_layout(
    title='Total Greenhouse Gas Emissions Over Time with AdaBoost Forecast',
    xaxis_title='Year',
    yaxis_title='Total Greenhouse Gas Emissions',
    legend=dict(x=0, y=1),
    hovermode='x unified'
)

# Plot the comparison
fig_combined = go.Figure()

# Actual Data
fig_combined.add_trace(go.Scatter(
    x=annual_data.index,
    y=annual_data.values,
    mode='lines',
    name='Actual Data',
    line=dict(color='red')
))

# ARIMA Forecast chart
fig_combined.add_trace(go.Scatter(
    x=forecast_index_arima,
    y=forecast_arima,
    mode='lines',
    name='ARIMA Forecast',
    line=dict(color='blue', dash='dash')
))

# Exponential Smoothing Forecast
fig_combined.add_trace(go.Scatter(
    x=forecast_index_es,
    y=forecast_es,
    mode='lines',
    name='Exponential Smoothing Forecast',
    line=dict(color='yellow', dash='dot')
))

# Polynomial Regression Forecast
fig_combined.add_trace(go.Scatter(
    x=future_predictions_poly['Year'],
    y=future_predictions_poly['Predicted_total_ghg'],
    mode='lines',
    name='Polynomial Regression Forecast',
    line=dict(color='green', dash='longdash')
))

# AdaBoost Forecast
fig_combined.add_trace(go.Scatter(
    x=future_predictions_ada['Year'],
    y=future_predictions_ada['Predicted_total_ghg'],
    mode='lines',
    name='AdaBoost Forecast',
    line=dict(color='purple', dash='dot')
))

# Update the layout
fig_combined.update_layout(
    title='Total Greenhouse Gas Emissions Forecast Comparison',
    xaxis_title='Year',
    yaxis_title='Total Greenhouse Gas Emissions',
    legend=dict(x=0, y=1),
    hovermode='x unified'
)

fig_errors = go.Figure()

# Calculate Mean Squared Errors
arima_mse = mean_squared_error(test, forecast_arima)
es_mse = mean_squared_error(test, forecast_es)
poly_mse = mean_squared_error(test, future_predictions_poly['Predicted_total_ghg'])
ada_mse = mean_squared_error(test, future_predictions_ada['Predicted_total_ghg'])

# Create the error comparison chart
fig_errors = go.Figure()

fig_errors.add_trace(go.Bar(
    x=['ARIMA'],
    y=[arima_mse],
    name='ARIMA MSE',
    marker_color='blue'
))

fig_errors.add_trace(go.Bar(
    x=['Exponential Smoothing'],
    y=[es_mse],
    name='Exponential Smoothing MSE',
    marker_color='red'
))

fig_errors.add_trace(go.Bar(
    x=['Polynomial Regression'],
    y=[poly_mse],
    name='Polynomial Regression MSE',
    marker_color='green'
))

fig_errors.add_trace(go.Bar(
    x=['AdaBoost'],
    y=[ada_mse],
    name='AdaBoost MSE',
    marker_color='purple'
))

fig_errors.update_layout(
    title='Model Error Comparison',
    xaxis_title='Model',
    yaxis_title='Error',
    barmode='group'
)

# Define the layout for the page
layout = html.Div(className='centered-content', children=[
    html.H1(children='Prediction models for Total Greenhouse Gas Emissions In Europe'),

    dcc.Graph(
        id='Arima-forecast-chart',
        figure=fig_arima
    ),

    dcc.Graph(
        id='Exponential smoothing forecast chart',
        figure=fig_es,
    ),

    dcc.Graph(
        id='AdaBoost forecast chart',
        figure=fig_ada
    ),

    dcc.Graph(
        id='Polynomial regression forecast chart',
        figure=fig_poly
    ),

    dcc.Graph(
        id='Comparison-chart',
        figure=fig_combined
    ),

dcc.Graph(
        id='error-comparison-chart',
        figure=fig_errors
    )
])