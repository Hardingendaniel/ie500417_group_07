from dash import dcc, html, Input, Output
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
from data.data import data, load_total_ghg_data, load_markdown

# Path to markdown files
ARIMA_MODEL_EXPLANATION = 'data/markdown/arima_model_explanation.md'
EXPONENTIAL_SMOOTHING_EXPLANATION = 'data/markdown/exponential_smoothing_explanation.md'
POLYNOMIAL_REGRESSION_EXPLANATION = 'data/markdown/polynomial_regression_explanation.md'
ADABOOST_REGRESSION_EXPLANATION = 'data/markdown/adaboost_regression_explanation.md'
COMPARISON_GRAPH_EXPLANATION = 'data/markdown/comparison_graph_explanation.md'
MODEL_ERROR_COMPARISON = 'data/markdown/model_error_comparison.md'

# List of European countries
european_countries = [
    "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium", "Bosnia and Herzegovina",
    "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Finland", "France", "Georgia", "Germany",
    "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Kazakhstan", "Kosovo", "Latvia", "Liechtenstein",
    "Lithuania", "Luxembourg", "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands", "North Macedonia",
    "Norway", "Poland", "Portugal", "Romania", "Russia", "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain",
    "Sweden", "Switzerland", "Turkey", "Ukraine", "United Kingdom", "Vatican City"
]

# The other dataset, dataset 2
total_ghg_data = load_total_ghg_data()
europe_ghg_data = total_ghg_data[total_ghg_data['Entity'].isin(european_countries)]
europe_ghg_by_year = europe_ghg_data.groupby('Year').sum().reset_index()


# Filter Data for European countries
europe_data = data[data['country'].isin(european_countries)].dropna(subset=["total_ghg", "year"]).copy()
europe_data['year'] = europe_data['year'].astype(int)
europe_data['Year'] = pd.to_datetime(europe_data['year'], format='%Y') + pd.offsets.YearEnd(0)
europe_data.set_index('Year', inplace=True)
europe_data['total_ghg'] = pd.to_numeric(europe_data['total_ghg'], errors='coerce')
europe_data = europe_data.dropna(subset=['total_ghg'])

# Resample Data to Annual Data
annual_data = europe_data['total_ghg'].resample('YE').sum()

# Align year ranges
all_years = pd.DataFrame({'Year': range(annual_data.index.min().year, 2023)})
europe_ghg_by_year = pd.merge(all_years, europe_ghg_by_year, on='Year', how='left')


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

# European Total GHG Emissions
fig_combined.add_trace(go.Scatter(
    x=europe_ghg_by_year['Year'],
    y=europe_ghg_by_year['Annual greenhouse gas emissions in CO₂ equivalents'],
    mode='lines',
    name='European Total GHG Emissions',
    line=dict(color='orange', dash='solid'),
    yaxis='y2'
))

# Update the layout
fig_combined.update_layout(
    title='Total Greenhouse Gas Emissions Forecast Comparison',
    xaxis_title='Year',
    yaxis_title='Total Greenhouse Gas Emissions (CO₂ Equivalents)',
    yaxis2=dict(
        title='European Total GHG Emissions',
        overlaying='y',
        side='right'
    ),
    legend=dict(x=0, y=1),
    hovermode='x unified'
)

fig_errors = go.Figure()

# Calculate Mean Squared Errors
arima_mse = mean_squared_error(test, forecast_arima)
es_mse = mean_squared_error(test, forecast_es)
poly_mse = mean_squared_error(test, future_predictions_poly['Predicted_total_ghg'])
ada_mse = mean_squared_error(test, future_predictions_ada['Predicted_total_ghg'])

# The other dataset, dataset 2
fig_total_ghg = go.Figure()

fig_total_ghg.add_trace(go.Scatter(
    x=europe_ghg_by_year['Year'],
    y=europe_ghg_by_year['Annual greenhouse gas emissions in CO₂ equivalents'],
    mode='lines',
    name='Total GHG Emissions',
    line=dict(color='red')
))

fig_total_ghg.update_layout(
    title='Total Greenhouse Gas Emissions Over Time for european countries',
    xaxis_title='Year',
    yaxis_title='Annual Greenhouse Gas Emissions (CO₂ Equivalents)',
    hovermode='x unified'
)

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

    dcc.Markdown(id='arima-model-explanation', dangerously_allow_html=True),
    dcc.Graph(id='Arima-forecast-chart', figure=fig_arima),

    dcc.Markdown(id='exponential-smoothing-explanation', dangerously_allow_html=True),
    dcc.Graph(id='Exponential smoothing forecast chart', figure=fig_es),

    dcc.Markdown(id='polynomial-regression-explanation', dangerously_allow_html=True),
    dcc.Graph(id='Polynomial regression forecast chart', figure=fig_poly),

    dcc.Markdown(id='adaboost-regression-explanation', dangerously_allow_html=True),
    dcc.Graph(id='AdaBoost forecast chart', figure=fig_ada),

    # Markdown for Comparison Graph Explanation
    dcc.Markdown(
        id='comparison-graph-explanation',
        dangerously_allow_html=True
    ),

    # Graph for Comparison Chart
    dcc.Graph(
        id='Comparison-chart',
        figure=fig_combined
    ),

    dcc.Markdown(id='model-error-comparison', dangerously_allow_html=True),
    dcc.Graph(id='model-error-comparison-chart', figure=fig_errors)

])
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
        Output('polynomial-regression-explanation', 'children'),
        Input('interval-component', 'n_intervals')
    )
    def update_polynomial_regression_explanation(n_intervals):
        return load_markdown(POLYNOMIAL_REGRESSION_EXPLANATION)

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
