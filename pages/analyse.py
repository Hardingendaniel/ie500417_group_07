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

    html.H3("Top 5 countries in different attributes"),

    # Year selector for the tables
    dcc.Dropdown(
        id='year-selector',
        options=[{'label': year, 'value': year} for year in range(1990, 2022)],
        value=1990,  # Default selected year
        clearable=False,
        style={'margin-bottom': '20px'}  # Add spacing below the dropdown
    ),

    # Container for the tables
    html.Div([
        # Table for displaying the 5 countries with highest co2_per_capita
        html.Div(id='highest-co2-table-container',
                 style={'display': 'inline-block', 'width': '32%', 'vertical-align': 'top'}),

        # Table for displaying the 5 countries with highest co2 emission
        html.Div(id='highest-co2-emission-table-container',
                 style={'display': 'inline-block', 'width': '32%', 'vertical-align': 'top'}),

        # Table for displaying the 5 countries with highest total_ghg
        html.Div(id='highest-total-ghg-table-container',
                 style={'display': 'inline-block', 'width': '32%', 'vertical-align': 'top'})
    ]),

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


# Callback to update the tables based on the selected year
@callback(
    [Output('highest-co2-table-container', 'children'),
     Output('highest-co2-emission-table-container', 'children'),
     Output('highest-total-ghg-table-container', 'children')],
    [Input('year-selector', 'value')]
)
def update_tables(selected_year):
    filtered_data = data[(data['year'] == selected_year) & (data['iso_code'].notna())]

    # Table for highest co2_per_capita
    highest_co2_per_capita = filtered_data.nlargest(5, 'co2_per_capita')[['country', 'co2_per_capita']]

    table_co2_per_capita = html.Table([
        html.Thead(html.Tr([html.Th(col) for col in highest_co2_per_capita.columns])),
        html.Tbody([
            html.Tr([html.Td(highest_co2_per_capita.iloc[i][col]) for col in highest_co2_per_capita.columns]) for i in
            range(len(highest_co2_per_capita))
        ])
    ], style={'border': '1px solid black', 'border-collapse': 'collapse', 'width': '100%'}),

    # Table for highest co2 emission
    highest_co2_emission = filtered_data.nlargest(5, 'co2')[['country', 'co2']]

    table_co2_emission = html.Table([
        html.Thead(html.Tr([html.Th(col) for col in highest_co2_emission.columns])),
        html.Tbody([
            html.Tr([html.Td(highest_co2_emission.iloc[i][col]) for col in highest_co2_emission.columns]) for i in
            range(len(highest_co2_emission))
        ])
    ], style={'border': '1px solid black', 'border-collapse': 'collapse', 'width': '100%'}),

    # Table for highest total_ghg
    highest_total_ghg = filtered_data.nlargest(5, 'total_ghg')[['country', 'total_ghg']]

    table_total_ghg = html.Table([
        html.Thead(html.Tr([html.Th(col) for col in highest_total_ghg.columns])),
        html.Tbody([
            html.Tr([html.Td(highest_total_ghg.iloc[i][col]) for col in highest_total_ghg.columns]) for i in
            range(len(highest_total_ghg))
        ])
    ], style={'border': '1px solid black', 'border-collapse': 'collapse', 'width': '100%'})

    return table_co2_per_capita, table_co2_emission, table_total_ghg
















