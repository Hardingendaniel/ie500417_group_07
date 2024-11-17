from dash import html, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px
from data.data import data

# Filter numeric columns for correlation calculation
numeric_df = data.select_dtypes(include='number')
columns_of_interest = ['co2', 'gdp', 'population', 'cumulative_co2', 'total_ghg', 'nitrous_oxide', 'methane']

# Page layout
layout = html.Div(
    className='centered-content',  # Apply centered layout
    children=[
        html.H1("Analysis with Data"),

        html.H1("Top 5 countries in different attributes"),

        # Year selector for the tables
        dcc.Dropdown(
            className='dropdown',
            id='year-selector',
            options=[{'label': year, 'value': year} for year in range(1990, 2022)],
            value=1990,  # Default selected year
            clearable=False,
        ),

        # Container for the tables
        html.Div(id='tables-container'),

        # Flex container for heatmap and checklist
        html.Div(
            className='heatmap-checklist-container',  # Use CSS for layout
            children=[
                # Graph for displaying the correlation matrix heatmap
                dcc.Graph(id='correlation-heatmap'),

                # Checklist for selecting columns
                dcc.Checklist(
                    id='attribute-checklist',
                    options=[{'label': col, 'value': col} for col in columns_of_interest],
                    value=columns_of_interest,  # Default selected columns
                    labelStyle={'margin-bottom': '5px'}  # Space between checklist items
                ),
            ]
        )
    ]
)

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
    Output('tables-container', 'children'),
    [Input('year-selector', 'value')]
)
def update_tables(selected_year):
    filtered_data = data[(data['year'] == selected_year) & (data['iso_code'].notna())]

    # Table for highest co2_per_capita
    highest_co2_per_capita = filtered_data.nlargest(5, 'co2_per_capita')[['country', 'co2_per_capita']]

    table_co2_per_capita = html.Table(
        className='information-tables',
        children=[
            html.Thead(html.Tr([html.Th(col) for col in highest_co2_per_capita.columns])),
            html.Tbody([
                html.Tr([html.Td(highest_co2_per_capita.iloc[i][col]) for col in highest_co2_per_capita.columns]) for i in range(len(highest_co2_per_capita))
            ])
        ]
    )

    # Table for highest co2 emission
    highest_co2_emission = filtered_data.nlargest(5, 'co2')[['country', 'co2']]

    table_co2_emission = html.Table(
        className='information-tables',
        children=[
            html.Thead(html.Tr([html.Th(col) for col in highest_co2_emission.columns])),
            html.Tbody([
                html.Tr([html.Td(highest_co2_emission.iloc[i][col]) for col in highest_co2_emission.columns]) for i in range(len(highest_co2_emission))
            ])
        ]
    )

    # Table for highest total_ghg
    highest_total_ghg = filtered_data.nlargest(5, 'total_ghg')[['country', 'total_ghg']]

    table_total_ghg = html.Table(
        className='information-tables',
        children=[
            html.Thead(html.Tr([html.Th(col) for col in highest_total_ghg.columns])),
            html.Tbody([
                html.Tr([html.Td(highest_total_ghg.iloc[i][col]) for col in highest_total_ghg.columns]) for i in range(len(highest_total_ghg))
            ])
        ]
    )

    # Wrap all tables in a div
    tables_div = html.Div(
        className='tables-container',
        children=[
            html.Div(table_co2_per_capita, className='information-tables'),
            html.Div(table_co2_emission, className='information-tables'),
            html.Div(table_total_ghg, className='information-tables'),
        ]
    )

    return tables_div





















