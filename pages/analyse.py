from dash import html, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px
from data.data import data

# Filter numeric columns for correlation calculation
numeric_df = data.select_dtypes(include='number')
columns_of_interest = ['total_ghg', 'co2', 'gdp', 'population', 'cumulative_co2',  'nitrous_oxide', 'methane']

# Page layout
layout = html.Div(
    className='centered-content',  # Apply centered layout
    children=[
        html.H1("Analysis with Data"),

        html.H1("Top 5 worst in different emission categories"),

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

        html.H1("Analytical factors leading to high total greenhouse emission (world and europe)"),

        html.Div(id='analytical_factors'),

        html.H1("Filtered Correlation Matrix Heatmap"),
        # Flex container for heatmap and checklist
        html.Div(
            className='heatmap-checklist-container',
            children=[
                html.Div(
                    className='heatmap-container',
                    children=[
                        dcc.Graph(id='correlation-heatmap')
                    ]
                ),
            html.Div(
                className='checklist-container',
                children=[
                    dcc.Checklist(
                        className='checklist',
                        id='attribute-checklist',
                        options=[{'label': col, 'value': col} for col in columns_of_interest],
                        value=columns_of_interest,  # Default selected columns
                    ),
                ]
            )
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
            width=600,
            height=600,
            margin=dict(l=20, r=20)
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
    highest_total_ghg = filtered_data.nlargest(5, 'total_ghg')[['country', 'total_ghg', 'ghg_per_capita']]

    table_total_ghg = html.Table(
        className='information-tables',
        children=[
            html.Thead(html.Tr([html.Th(col) for col in highest_total_ghg.columns])),
            html.Tbody([
                html.Tr([html.Td(highest_total_ghg.iloc[i][col]) for col in highest_total_ghg.columns])
                for i in range(len(highest_total_ghg))
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

# Callback to update the analytical factors based on the selected year
@callback(
    Output('analytical_factors', 'children'),
    [Input('year-selector', 'value')]
)
def update_table_analytical(selected_year):
    # European countries ISO codes
    europe = ['NOR', 'SWE', 'FIN', 'DNK', 'ISL', 'GBR', 'IRL', 'DEU', 'NLD', 'POL',
              'EST', 'LVA', 'LTU', 'RUS', 'BLR', 'CZE', 'SVK', 'FRA', 'UKR',
              'CHE', 'AUT', 'HUN', 'BEL', 'LUX', 'MDA', 'ITA', 'ESP', 'PRT', 'GRC', 'MLT',
              'CYP', 'HRV', 'SVN', 'BGR', 'ROU', 'ALB', 'BIH', 'MNE', 'MKD', 'MCO', 'SMR',
              'VAT', 'AND', 'SRB']

    # Make copy of the filtered data to avoid SettingWithCopyWarning
    filtered_data2 = data[(data['year'] == selected_year) & (data['iso_code'].notna())].copy()

    # Table for highest total_ghg (Global)
    highest_total_ghg_anal = filtered_data2.nlargest(5, 'total_ghg')[['country', 'total_ghg']]

    # Population and GDP columns
    highest_total_ghg_anal = highest_total_ghg_anal.merge(
        filtered_data2[['country', 'population', 'gdp']],
        on='country',
    )

    # Compute population and GDP ranks (Global)
    filtered_data2['pop_rank'] = filtered_data2['population'].rank(method='min', ascending=False)
    filtered_data2['gdp_rank'] = filtered_data2['gdp'].rank(method='min', ascending=False)

    # Map ranks to the top GHG countries (Global)
    highest_total_ghg_anal['Population_with_rank'] = highest_total_ghg_anal.apply(
        lambda
            row: f"{row['population']:,} ({int(filtered_data2[filtered_data2['country'] == row['country']]['pop_rank'].values[0])})",
        axis=1
    )
    highest_total_ghg_anal['GDP_with_rank'] = highest_total_ghg_anal.apply(
        lambda
            row: f"{row['gdp']:,} ({int(filtered_data2[filtered_data2['country'] == row['country']]['gdp_rank'].values[0])})",
        axis=1
    )

    # Filter European countries
    europe_data = filtered_data2[filtered_data2['iso_code'].isin(europe)].copy()

    # Compute population and GDP ranks for European subset
    europe_data['pop_rank'] = europe_data['population'].rank(method='min', ascending=False)
    europe_data['gdp_rank'] = europe_data['gdp'].rank(method='min', ascending=False)

    # Table for highest total_ghg (Europe)
    highest_europe_total_ghg_anal = europe_data.nlargest(5, 'total_ghg')[['country', 'total_ghg']]
    highest_europe_total_ghg_anal = highest_europe_total_ghg_anal.merge(
        europe_data[['country', 'population', 'gdp']],
        on='country',
    )

    # Map ranks to the top GHG countries (Europe)
    highest_europe_total_ghg_anal['Population_with_rank'] = highest_europe_total_ghg_anal.apply(
        lambda
            row: f"{row['population']:,} ({int(europe_data[europe_data['country'] == row['country']]['pop_rank'].values[0])})",
        axis=1
    )
    highest_europe_total_ghg_anal['GDP_with_rank'] = highest_europe_total_ghg_anal.apply(
        lambda
            row: f"{row['gdp']:,} ({int(europe_data[europe_data['country'] == row['country']]['gdp_rank'].values[0])})",
        axis=1
    )

    # Prepare the global HTML table
    table_headers = ['Country', 'Total GHG', 'Population (Rank)', 'GDP (Rank)']
    table_rows = [
        html.Tr([
            html.Td(row['country']),
            html.Td(f"{row['total_ghg']:,}"),
            html.Td(row['Population_with_rank']),
            html.Td(row['GDP_with_rank'])
        ])
        for _, row in highest_total_ghg_anal.iterrows()
    ]

    table_total_ghg_analytical = html.Table(
        className='information-tables',
        children=[
            html.Thead(html.Tr([html.Th(header) for header in table_headers])),
            html.Tbody(table_rows)
        ]
    )

    # Prepare the European HTML table
    europe_table_rows = [
        html.Tr([
            html.Td(row['country']),
            html.Td(f"{row['total_ghg']:,}"),
            html.Td(row['Population_with_rank']),
            html.Td(row['GDP_with_rank'])
        ])
        for _, row in highest_europe_total_ghg_anal.iterrows()
    ]

    table_total_ghg_europe = html.Table(
        className='information-tables',
        children=[
            html.Thead(html.Tr([html.Th(header) for header in table_headers])),
            html.Tbody(europe_table_rows)
        ]
    )

    # Combine both tables into the div with padding
    analytical_div = html.Div(
        className='analytical_factors',
        children=[
            html.Div(table_total_ghg_analytical, className='information-tables'),
            html.Div(
                table_total_ghg_europe,
                className='information-tables',
                style={'marginTop': '20px'}  # Add padding between the tables
            ),
        ]
    )

    return analytical_div























