# pages/analyse.py
from dash import html, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px
from data.data import data

# Filter numeric columns for correlation calculation
numeric_df = data.select_dtypes(include='number')
columns_of_interest = ['co2', 'gdp', 'population', 'cumulative_co2', 'total_ghg', 'nitrous_oxide', 'methane']

# Page layout
layout = html.Div([
    html.H2("Analysis with Data"),

    # Checklist for selecting columns
    dcc.Checklist(
        id='attribute-checklist',
        options=[{'label': col, 'value': col} for col in columns_of_interest],
        value=columns_of_interest,  # Default selected columns
        labelStyle={'display': 'inline-block', 'margin-right': '10px'}
    ),

    # Graph for displaying the correlation matrix heatmap
    dcc.Graph(id='correlation-heatmap')
])

# Callback to update the heatmap based on selected attributes
@callback(
    Output('correlation-heatmap', 'figure'),
    [Input('attribute-checklist', 'value')]
)
def update_heatmap(selected_columns):
    # Filter the correlation matrix based on selected columns
    if selected_columns:
        correlation_matrix_filtered = numeric_df[selected_columns].corr()
        fig = px.imshow(
            correlation_matrix_filtered,
            text_auto=True,
            labels=dict(color='Correlation')
        )
        fig.update_layout(
            title='Filtered Correlation Matrix Heatmap',
            width=800,
            height=600
        )
        fig.update_xaxes(tickangle=45)
        return fig
    return {}  # Return an empty figure if no columns are selected



