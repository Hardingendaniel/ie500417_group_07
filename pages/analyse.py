from dash import html, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px
from data.data import data, load_markdown, load_ghg_release, load_gh_by_sector

# Filter numeric columns for correlation calculation
numeric_df = data.select_dtypes(include='number')
columns_of_interest = ['total_ghg', 'co2', 'gdp', 'population', 'cumulative_co2',  'nitrous_oxide', 'methane']

# First paragraph
MARKDOWN_FILE_PATH = 'data/markdown/explanation_of_ghg.md'
# Second paragraph
MARKDOWN_FILE_PATH2 = 'data/markdown/where_ghg_comes_from.md'

# Define layout with Tabs
layout = html.Div([
    dcc.Tabs([
        dcc.Tab(className='tabs-title', label='Greenhouse Gases', children=[
            html.Div(
                className='centered-content',
                children=[
                    html.H1("Explanation of Greenhouse Gases"),
                    # Markdown content
                    html.Div(
                        dcc.Markdown(
                            id='markdown-content-tab2',
                            dangerously_allow_html=True
                        ),
                        className='centered-content markdown-content',
                    ),
                    # Stacked area chart for greenhouse gases
                    html.Div(
                        dcc.Graph(
                            id='ghg-stacked-chart'
                        ),
                        className='centered-content'
                    ),
                    html.Div(
                        [
                            html.H2("Human Impact on the Greenhouse Effect", style={'textAlign': 'center'}),
                            dcc.Markdown(
                                id='markdown-content-tab2-p2',
                                dangerously_allow_html=True

                            )
                        ],
                        className='centered-content markdown-content'
                    ),
                    # The GHG per sector plot's div
                    html.Div(
                        className='centered-content',
                        children=[
                            # Dropdown selector which defaults to 'World' (due to the filter farther down.)
                            html.Div([
                                html.H2("Select Region"),
                                dcc.Dropdown(
                                    id='region-selector-category',
                                    options=[
                                        {'label': 'Africa', 'value': 'Africa'},
                                        {'label': 'Asia', 'value': 'Asia'},
                                        {'label': 'Europe', 'value': 'Europe'},
                                        {'label': 'North America', 'value': 'North America'},
                                        {'label': 'Oceania', 'value': 'Oceania'},
                                        {'label': 'South America', 'value': 'South America'},
                                        {'label': 'World', 'value': 'World'}
                                    ],
                                    value='Europe',  # Default to 'Europe'
                                    clearable=False
                                )
                            ], style={'margin-bottom': '20px'}),

                            # Graph for category emissions
                            dcc.Graph(
                                id='ghg-category-plot'
                            )
                        ]
                    ),

                ]
            )
        ]),


        # Tab 2:Analysis with Data
        dcc.Tab(className='tabs-title', label='Analysis with Data', children=[
            html.Div(
                className='centered-content',
                children=[

                    html.H1("Top 5 Worst in Different Emission Categories"),
                    dcc.Dropdown(
                        className='dropdown',
                        id='year-selector',
                        options=[{'label': year, 'value': year} for year in range(1990, 2022)],
                        value=1990,
                        clearable=False,
                    ),
                    html.Div(id='tables-container'),

                    html.H1("Analytical Factors Leading to High Total Greenhouse Emission (World and Europe)"),
                    html.Div(id='analytical_factors'),

                    html.H1("Filtered Correlation Matrix Heatmap"),
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
                                        options=[{'label': col, 'value': col} for col in ['total_ghg', 'co2', 'gdp', 'population', 'cumulative_co2', 'nitrous_oxide', 'methane']],
                                        value=['total_ghg', 'co2', 'gdp', 'population', 'cumulative_co2', 'nitrous_oxide', 'methane'],
                                    )
                                ]
                            )
                        ]
                    )
                ]
            )
        ]),
    ])
])

def init_callbacksalso(app):
    """Register callbacks for Analyse."""

    @app.callback(
        Output('markdown-content-tab2', 'children'),
        Input('interval-component', 'n_intervals'))
    def update_markdown(n_intervals):
        return load_markdown(MARKDOWN_FILE_PATH)
    @app.callback(
        Output('markdown-content-tab2-p2', 'children'),
        Input('interval-component', 'n_intervals'))
    def update_markdown2(n_intervals):
        return load_markdown(MARKDOWN_FILE_PATH2)

    @app.callback(
        Output('ghg-stacked-chart', 'figure'),
        Input('interval-component', 'n_intervals')
    )
    def update_ghg_chart(n_intervals):
        """Plots the greenhouse gas emission % which shows the % of how much each gas is releasing compared to the total"""
        df = load_ghg_release()

        gas_columns = [
            'Annual CO₂ emissions',
            'Annual methane emissions in CO₂ equivalents',
            'Annual nitrous oxide emissions in CO₂ equivalents'
        ]
        df[gas_columns] = df[gas_columns].fillna(0).clip(lower=0)

        df_grouped = df.groupby('Year')[gas_columns].sum().reset_index()

        # Calculate the percentages of each type of gas.
        df_grouped['Total'] = df_grouped[gas_columns].sum(axis=1)
        for gas in gas_columns:
            df_grouped[gas + ' (%)'] = (df_grouped[gas] / df_grouped['Total']) * 100

        # Restructure the data for the stacked chart.
        df_long = pd.melt(
            df_grouped,
            id_vars=['Year'],
            value_vars=[col + ' (%)' for col in gas_columns],
            var_name='Gas Type',
            value_name='Percentage'
        )

        # Renaming the column names to be less obtuse.
        df_long['Gas Type'] = df_long['Gas Type'].replace({
            'Annual CO₂ emissions (%)': 'CO₂',
            'Annual methane emissions in CO₂ equivalents (%)': 'Methane',
            'Annual nitrous oxide emissions in CO₂ equivalents (%)': 'Nitrous'
        })

        # Stacked area chart, with co2 on the bottom
        fig = px.area(
            df_long,
            x='Year',
            y='Percentage',
            color='Gas Type',
            labels={
                'Percentage': 'Percentage (%)',
                'Year': 'Year',
                'Gas Type': 'Gas Type'
            },
            template='plotly_white',
            hover_data={
                'Percentage': ':.2f',
                'Year': True,
                'Gas Type': True
            }
        )

        # Update hover template to show what info I want specifically
        fig.update_traces(
            hoveron='points',  # why doesn't fills work :()
            hovertemplate=(
                "<b>Gas Type:</b> %{customdata[0]}<br>"
                "<b>Year:</b> %{x}<br>"
                "<b>Percentage:</b> %{y:.2f}%<br>"
                "<extra></extra>"
            ),
            mode='none'
        )

        # Update layout
        fig.update_layout(
            xaxis=dict(
                title='Year',
                tickvals=list(range(1870, 2020, 20)) + [2023] + [1855],  # 2023 included, 2020 removed, else 2020 was on top of 2023 lol
                range=[1850, 2023], # actual range
            ),
            yaxis=dict(
                title='Percentage (%)',
                ticksuffix='%'
            ),
            legend=dict(title='Gas Type', traceorder='reversed'),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20)
        )


        return fig

    @app.callback(
        Output('ghg-category-plot', 'figure'),
        [Input('region-selector-category', 'value'), Input('interval-component', 'n_intervals')]
    )
    def update_category_plot(selected_region, n_intervals):
        df = load_gh_by_sector()

        # Filter data by region
        if selected_region != 'World':
            df = df[df['Entity'] == selected_region]

        category_columns = [
            'Greenhouse gas emissions from agriculture',
            'Greenhouse gas emissions from land use change and forestry',
            'Greenhouse gas emissions from waste',
            'Greenhouse gas emissions from buildings',
            'Greenhouse gas emissions from industry',
            'Greenhouse gas emissions from manufacturing and construction',
            'Greenhouse gas emissions from transport',
            'Greenhouse gas emissions from electricity and heat',
            'Fugitive emissions of greenhouse gases from energy production',
            'Greenhouse gas emissions from other fuel combustion',
            'Greenhouse gas emissions from bunker fuels'
        ]

        df_grouped = df.groupby('Year')[category_columns].sum().reset_index()

        latest_year = df_grouped['Year'].max()

        # Get total emissions for each category in the latest year
        latest_emissions = (
            df_grouped[df_grouped['Year'] == latest_year]
            .melt(value_vars=category_columns, var_name='Category', value_name='Emissions')
            .sort_values(by='Emissions', ascending=False)
        )

        sorted_categories = latest_emissions['Category'].tolist()

        df_long = pd.melt(
            df_grouped,
            id_vars=['Year'],
            value_vars=category_columns,
            var_name='Category',
            value_name='Emissions'
        )

        df_long['Category'] = df_long['Category'].str.replace(
            "Greenhouse gas emissions from ", "", regex=False
        ).str.replace(
            "Fugitive emissions of greenhouse gases from ", "Fugitive emissions from ", regex=False
        ).str.lower().str.capitalize()

        # Update category orders to capitalize the first word.
        sorted_categories_simple = [
            category.replace("Greenhouse gas emissions from ", "").replace(
                "Fugitive emissions of greenhouse gases from ", "Fugitive emissions from "
            ).lower().capitalize()
            for category in sorted_categories
        ]

        fig = px.line(
            df_long,
            x='Year',
            y='Emissions',
            color='Category',
            labels={
                'Emissions': 'Emissions (kt CO₂e)',
                'Year': 'Year',
                'Category': 'Sector'
            },
            category_orders={'Category': sorted_categories_simple}
        )

        fig.update_layout(
            template='plotly_white',
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            legend_title=dict(
                text="Greenhouse gas emissions by sector"
            )
        )

        return fig



    # Callback to update the heatmap based on selected attributes
    @app.callback(
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
    @app.callback(
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
    @app.callback(
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
























