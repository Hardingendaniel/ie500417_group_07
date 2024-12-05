from dash import html, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px
from data.data import data, load_markdown, load_ghg_release, load_gh_by_sector
from .analytical_operations import AnalyticalOperations

# First paragraph
MARKDOWN_FILE_PATH = 'data/markdown/explanation_of_ghg.md'
# Second paragraph
MARKDOWN_FILE_PATH2 = 'data/markdown/where_ghg_comes_from.md'

MARKDOWN_FILE_PATH3 = 'data/markdown/top5_barchart.md'

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
                    html.Div(
                        [
                            html.H1("Top 5 Worst in Different Emission Categories"),
                            dcc.Markdown(
                                id='markdown-content-tab2-p3',
                                dangerously_allow_html=True

                            )
                        ],
                        className='centered-content markdown-content'
                    ),

                    dcc.Dropdown(
                        className='dropdown',
                        id='year-selector',
                        options=[{'label': year, 'value': year} for year in range(1990, 2021)],
                        value=2019,
                        clearable=False,
                    ),

                    html.Div(id='charts-container'),

                    html.H1("Analytical Factors Leading to High Total Greenhouse Emission (World and Europe)"),
                    html.Div(id='analytical_factors'),
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
        Output('markdown-content-tab2-p3', 'children'),
        Input('interval-component', 'n_intervals'))
    def update_markdown2(n_intervals):
        return load_markdown(MARKDOWN_FILE_PATH3)

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

        # Update layout with the source annotation
        fig.update_layout(
            xaxis=dict(
                title='Year',
                tickvals=list(range(1870, 2020, 20)) + [2023] + [1855],  # 2023 included, 2020 removed, else 2020 was on top of 2023.
                range=[1850, 2023],  # Actual range
                fixedrange=True  # Prevent zooming out, Daniel found this 😎
            ),
            yaxis=dict(
                title='Percentage (%)',
                ticksuffix='%'
            ),
            legend=dict(title='Gas Type', traceorder='reversed'),
            height=400,
            margin=dict(l=0, r=0, t=0, b=80),
            annotations=[
                dict(
                    text="Source for dataset: https://ourworldindata.org/grapher/ghg-emissions-by-gas",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.2,
                    xanchor='center',
                    yanchor='top',
                    font=dict(size=12, color="gray")
                )
            ]
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
            height=500,
            margin=dict(l=0, r=0, t=0, b=100),
            legend_title=dict(
                text="Greenhouse gas emissions by sector"
            ),
            annotations=[
                dict(
                    text="Source for dataset: https://ourworldindata.org/grapher/ghg-emissions-by-sector",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.15,
                    xanchor='center',
                    yanchor='top',
                    font=dict(size=12, color="gray")
                )
        ]
        )

        return fig

    # Callback to update the tables based on the selected year
    @app.callback(
        Output('charts-container', 'children'),
        [Input('year-selector', 'value')]
    )
    def update_charts(selected_year):
        filtered_data = data[(data['year'] == selected_year) & (data['iso_code'].notna())]

        # Bar chart for highest co2_per_capita
        highest_co2_per_capita = filtered_data.nlargest(5, 'co2_per_capita')[['country', 'co2_per_capita']]
        bar_chart_co2_per_capita = px.bar(
            highest_co2_per_capita,
            x='country',
            y='co2_per_capita',
            title='Top 5 Countries by CO2 per Capita'
        )
        bar_chart_co2_per_capita.update_yaxes(range=[0, None], fixedrange=True)

        # Bar chart for highest co2 emission
        highest_co2_emission = filtered_data.nlargest(5, 'co2')[['country', 'co2']]
        bar_chart_co2_emission = px.bar(
            highest_co2_emission,
            x='country',
            y='co2',
            title='Top 5 Countries by CO2 Emission'
        )
        bar_chart_co2_emission.update_yaxes(range=[0, None], fixedrange=True)

        # Bar chart for highest total_ghg
        highest_total_ghg = filtered_data.nlargest(5, 'total_ghg')[['country', 'total_ghg', 'ghg_per_capita']]
        bar_chart_total_ghg = px.bar(
            highest_total_ghg,
            x='country',
            y='total_ghg',
            title='Top 5 Countries by Total GHG'
        )
        bar_chart_total_ghg.update_yaxes(range=[0, None], fixedrange=True)

        # Wrap all charts in a div
        charts_div = html.Div(
            className='charts-container',
            children=[
                dcc.Graph(figure=bar_chart_co2_per_capita, style={'height': '450px', 'width': '450px'}),
                dcc.Graph(figure=bar_chart_co2_emission, style={'height': '450px', 'width': '450px'}),
                dcc.Graph(figure=bar_chart_total_ghg, style={'height': '450px', 'width': '450px'}),
            ]
        )

        return charts_div

    #European countries ISO codes
    europe = ['NOR', 'SWE', 'FIN', 'DNK', 'ISL', 'GBR', 'IRL', 'DEU', 'NLD', 'POL',
              'EST', 'LVA', 'LTU', 'RUS', 'BLR', 'CZE', 'SVK', 'FRA', 'UKR', 'CHE',
              'AUT', 'HUN', 'BEL', 'LUX', 'MDA', 'ITA', 'ESP', 'PRT', 'GRC', 'MLT',
              'CYP', 'HRV', 'SVN', 'BGR', 'ROU', 'ALB', 'BIH', 'MNE', 'MKD', 'MCO',
              'SMR', 'VAT', 'AND', 'SRB']


    analytical_ops = AnalyticalOperations(data, europe)

    @app.callback(
        Output('analytical_factors', 'children'),
        [Input('year-selector', 'value')]
    )
    def update_table_analytical(selected_year):

        analytical_div = analytical_ops.compute_analytical_factors(selected_year)

        return analytical_div


















