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
    highest_total_ghg_anal = filtered_data2.nlargest(5, 'total_ghg')[['country', 'total_ghg']]
    highest_total_ghg_anal = highest_total_ghg_anal.merge(
        filtered_data2[['country', 'population', 'gdp']],
        on='country',
    )

    # Compute ranks
    filtered_data2['pop_rank'] = filtered_data2['population'].rank(method='min', ascending=False)
    filtered_data2['gdp_rank'] = filtered_data2['gdp'].rank(method='min', ascending=False)

    # Map ranks to the top GHG countries (Global)
    highest_total_ghg_anal['Population_with_rank'] = highest_total_ghg_anal.apply(
        lambda row: f"{row['population']:,} ({int(filtered_data2[filtered_data2['country'] == row['country']]['pop_rank'].values[0])})",
        axis=1
    )
    highest_total_ghg_anal['GDP_with_rank'] = highest_total_ghg_anal.apply(
        lambda row: int(filtered_data2[filtered_data2['country'] == row['country']]['gdp_rank'].values[0]),
        axis=1
    )

    # Compute GHG emissions for rest of the world
    rest_of_world_ghg = filtered_data2['total_ghg'].sum() - highest_total_ghg_anal['total_ghg'].sum()

    # Prepare data for the global pie chart
    global_pie_data = highest_total_ghg_anal[['country', 'total_ghg']].copy()

    top_5_world_ghg = highest_total_ghg_anal[['country', 'total_ghg', 'population']].copy()
    rest_of_world_population = filtered_data2['population'].sum() - top_5_world_ghg['population'].sum()
    world_pie_pop_data = pd.concat([
        top_5_world_ghg[['country', 'population']],
        pd.DataFrame({'country': ['Rest of World'], 'population': [rest_of_world_population]})
    ], ignore_index=True)


    # Add 'Rest of the World' row
    global_pie_data = pd.concat([
        global_pie_data,
        pd.DataFrame({'country': ['Rest of the World'], 'total_ghg': [rest_of_world_ghg]})
    ], ignore_index=True)


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
        lambda row: f"{row['population']:,} ({int(europe_data[europe_data['country'] == row['country']]['pop_rank'].values[0])})",
        axis=1
    )
    highest_europe_total_ghg_anal['GDP_with_rank'] = highest_europe_total_ghg_anal.apply(
        lambda row: int(europe_data[europe_data['country'] == row['country']]['gdp_rank'].values[0]),
        axis=1
    )

    rest_of_europe_ghg = europe_data['total_ghg'].sum() - highest_europe_total_ghg_anal['total_ghg'].sum()
    europe_pie_data = highest_europe_total_ghg_anal[['country', 'total_ghg']].copy()
    europe_pie_data = pd.concat([
        europe_pie_data,
        pd.DataFrame({'country': ['Rest of Europe'], 'total_ghg': [rest_of_europe_ghg]})
    ], ignore_index=True)

    top_5_europe_ghg_countries = highest_europe_total_ghg_anal[['country', 'total_ghg', 'population']].copy()
    rest_of_europe_population = europe_data['population'].sum() - top_5_europe_ghg_countries['population'].sum()
    europe_pie_pop_data = pd.concat([
        top_5_europe_ghg_countries[['country', 'population']],
        pd.DataFrame({'country' : ['Rest of Europe'], 'population': [rest_of_europe_population]})
    ], ignore_index=True)


    # Create the bar chart for total GHG emissions
    fig = px.bar(
        highest_total_ghg_anal,
        x='country',
        y='total_ghg',
        hover_data={'total_ghg': False, 'Population_with_rank': True, 'GDP_with_rank': True},
        labels={'total_ghg': 'Total GHG Emissions', 'country': 'Country'},
        title='Top 5 Countries by Total GHG Emissions'
    )

    fig.update_traces(hovertemplate='<br>'.join([
        'Country: %{x}',
        'Total GHG Emissions: %{y}',
        'Population (Rank): %{customdata[0]}',
        'GDP Rank: %{customdata[1]}'
    ]))

    bar_chart = dcc.Graph(
        figure=fig
    )

    # Create the bar chart for total GHG emissions
    fig_europe = px.bar(
        highest_europe_total_ghg_anal,
        x='country',
        y='total_ghg',
        hover_data={'total_ghg': False, 'Population_with_rank': True, 'GDP_with_rank': True},
        labels={'total_ghg': 'Total GHG Emissions', 'country': 'Country'},
        title='Top 5 Countries by Total GHG Emissions in Europe'
    )

    fig.update_traces(hovertemplate='<br>'.join([
        'Country: %{x}',
        'Total GHG Emissions: %{y}',
        'Population (Rank): %{customdata[0]}',
        'GDP Rank: %{customdata[1]}'
    ]))

    bar_chart_europe = dcc.Graph(
        figure=fig_europe
    )

    # Create the global pie chart
    global_pie_fig = px.pie(
        global_pie_data,
        names='country',
        values='total_ghg',
        title='GHG Emissions Distribution (Top 5 Countries vs Rest of World)'
    )

    global_pie_chart = dcc.Graph(
        figure=global_pie_fig
    )

    world_pop_fig = px.pie(
        world_pie_pop_data,
        names='country',
        values='population',
        title='Respective countries Population Distribution (Top 5 vs Rest of Europe)'
    )
    world_pop_chart = dcc.Graph(figure=world_pop_fig)

    #HER MÅ EG ADDE CSS STYLING i egen css fil; MEN FOR LAT TIL DET NO
    world_section = html.Div(
        style={'display': 'flex', 'justifyContent': 'space-around', 'align-items': 'center', 'margin-bottom': '20px'},
        children=[
            html.Div(global_pie_chart, style={'flex': '1', 'padding': '10px'}),
            html.Div(world_pop_chart, style={'flex': '1', 'padding': '10px'})
        ]
    )

    # Create the global pie chart
    europe_pie_fig = px.pie(
        europe_pie_data,
        names='country',
        values='total_ghg',
        title='GHG Emissions Distribution (Top 5 European vs Rest of World)'
    )

    europe_pie_chart = dcc.Graph(
        figure=europe_pie_fig
    )

    # Create European Population pie chart
    europe_pop_fig = px.pie(
        europe_pie_pop_data,
        names='country',
        values='population',
        title='Respective countries Population Distribution (Top 5 vs Rest of Europe)'
    )
    europe_pop_chart = dcc.Graph(figure=europe_pop_fig)

    #HER MÅ EG ADDE CSS STYLING i egen css fil; MEN FOR LAT TIL DET NO
    europe_section = html.Div(
        style={'display': 'flex', 'justifyContent': 'space-around', 'align-items': 'center', 'margin-bottom': '20px'},
        children=[
            html.Div(europe_pie_chart, style={'flex': '1', 'padding': '10px'}),
            html.Div(europe_pop_chart, style={'flex': '1', 'padding': '10px'})
        ]
    )

    # Combine both tables and the bar chart into the div with padding
    analytical_div = html.Div(
        className='analytical_factors',
        children=[
            bar_chart,
            world_section,
            bar_chart_europe,
            europe_section,
        ]
    )

    return analytical_div
























