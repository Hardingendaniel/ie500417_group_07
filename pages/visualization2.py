from dash import html, dcc, Input, Output
from data.data import load_markdown, ghg_data, global_temp_data
import pandasql as sql
import pandas as pd
import plotly.express as px


# Path to the markdown file
MARKDOWN_FILE_PATH = 'data/winter_1995_96.md'

# Queries
def query_country_data(ghg_data):
    """Run the query to extract total GHG emissions by country."""
    query = """
    SELECT DISTINCT Code AS iso_code, Entity AS country, Year AS year,
                    "Annual greenhouse gas emissions in CO₂ equivalents" AS total_ghg
    FROM ghg_data
    WHERE Code IS NOT NULL AND Code != "" AND LENGTH(Code) = 3
    GROUP BY Code, Year
    """
    return sql.sqldf(query, locals())

# Classify the North / South European countries
def classify_european_countries(country_result):
    """Classify countries into Northern or Southern Europe."""
    northern_europe = [
        'NOR', 'SWE', 'FIN', 'DNK', 'ISL', 'GBR', 'IRL', 'DEU', 'NLD', 'POL', 
        'EST', 'LVA', 'LTU', 'RUS', 'BLR', 'CZE', 'SVK', 'FRA', 'UKR', 
        'CHE', 'AUT', 'HUN', 'BEL', 'LUX', 'MDA'
    ]
    southern_europe = [
        'ITA', 'ESP', 'PRT', 'GRC', 'MLT', 'CYP', 'HRV', 'SVN', 'BGR', 
        'ROU', 'ALB', 'BIH', 'MNE', 'MKD', 'MCO', 'SMR', 'VAT', 'AND', 'SRB'
    ]

    # Add a 'north_or_south' column
    country_result['north_or_south'] = country_result['iso_code'].apply(
        lambda x: 'Northern' if x in northern_europe else ('Southern' if x in southern_europe else None)
    )
    return country_result[country_result['north_or_south'].notnull()]

# Query for grabbing the temperatures from selected countries and years.
def query_selected_country_temperatures(global_temp_data):
    """
    Extract temperature data from the selected countries.
    from the years 1995 to 1997, ignoring the 'Year' column which represents months, those are mapped further down.
    """
    # Only look at the Northern countries which were affected by the temperature changes, more so than more southern countries.
    desired_codes = ['NOR', 'SWE', 'FIN', 'DNK', 'ISL', 'GBR', 'IRL', 'DEU', 'NLD', 'POL', 
        'EST', 'LVA', 'LTU', 'RUS', 'BLR', 'CZE', 'SVK', 'FRA', 'UKR', 
        'CHE', 'AUT', 'HUN', 'BEL', 'LUX', 'MDA']

    # Define the years we are looking at.
    selected_years = ['1995', '1996', '1997']

    # Use backticks for year columns since they are numeric and may cause syntax issues
    columns_to_select = ['Entity', 'Code', 'Year'] + selected_years
    columns_str = ', '.join([f'`{col}`' for col in columns_to_select])

    # Define the query to grab the countries and the "years".
    query = f"""
    SELECT {columns_str}
    FROM global_temp_data
    WHERE Code IN ({', '.join(['"' + code + '"' for code in desired_codes])})
    """

    result = sql.sqldf(query, locals())

    return result

# Apply the queries
# Global temperatures in span of years chosen.
global_temps = query_selected_country_temperatures(global_temp_data)
# total_ghg from 1990 to 2022.
country_result = query_country_data(ghg_data)
#Classify countries into North- or Southern Europe.
europe_result = classify_european_countries(country_result)

# Layout for Visualization with Tabs
layout = html.Div([

    dcc.Tabs([

        # Tab 1: Total GHG Emission In Europe, Map
        dcc.Tab(label='Map of Europe', children=[
            html.Div(
                [
                    html.H2('Analysis for the Total Greenhouse Gas Emission in Europe'),
                    dcc.Graph(id='choropleth-map'),
                    dcc.Slider(
                        id='year-slider',
                        min=1990,
                        max=2022,
                        value=1990,
                        marks={str(year): str(year) for year in range(1990, 2023)},
                        step=1,  # Enables finer granularity for sliding
                        tooltip={"placement": "bottom", "always_visible": True},
                        updatemode='drag'  # Critical for real-time updates
                    ),
                    html.Div(id='selected-data', className='centered-content')
                ],
                className='centered-content'  # Apply CSS class
            )
        ]),

        # Tab 2: Winter of 1995/96
        dcc.Tab(label='Winter of 1995/96', children=[
            html.Div(
                [
                    # Markdown content at the top
                    html.Div(
                        dcc.Markdown(
                            id='markdown-content',
                            dangerously_allow_html=True
                        ),
                        className='centered-content',
                        style={"margin-bottom": "20px"}
                    ),

                    # Aggregated countries, plotted 95-98
                    html.Div(
                        dcc.Graph(
                            id='temp-aggregated-1995-1997-graph',
                            style={"margin-bottom": "20px"}
                        ),
                        className='centered-content'
                    ),

                    # Existing GHG Aggregated Trend Graph
                    html.Div(
                        dcc.Graph(
                            id='ghg-aggregated-trend-graph'
                        ),
                        className='centered-content'
                    )
                ],
                className='centered-content'
            )
        ]),

    ]),

    # Interval component to refresh data every 30 seconds, just for testing will be removed by delivery.
    dcc.Interval(
        id='interval-component',
        interval=30 * 1000,  # 30 seconds in milliseconds
        n_intervals=0
    )
])

def init_callbacks(app):
    """Register callbacks for visualization2."""
    
    # Callback for the choropleth map in the first tab
    @app.callback(
        Output('choropleth-map', 'figure'),
        Input('year-slider', 'value')  # Triggered continuously as the slider is dragged
    )
    def update_map(selected_year):
        # Filter the data for the selected year
        filtered_df = europe_result[europe_result['year'] == selected_year].copy()
        
        # Rename columns for user readability
        filtered_df = filtered_df.rename(columns={"total_ghg": "Total Greenhouse Gas Emission"})

        # Create the choropleth map
        fig = px.choropleth(
            filtered_df,
            locations="iso_code",
            color="Total Greenhouse Gas Emission",
            hover_name="country",
            color_continuous_scale=[
                (0.0, "#ffffe0"),  # Light yellow
                (0.2, "#ffd59b"),  # Light orange
                (0.4, "#fdae61"),  # Orange
                (0.6, "#f46d43"),  # Red-orange
                (0.8, "#d73027"),  # Red
                (1.0, "#a50026")   # Dark red
            ],
            range_color=(0, 4e9),  # Set range from 0 to 4 billion
            scope="europe"  # Focus on Europe only
        )

        # Configure the map layout
        fig.update_geos(
            projection_type="natural earth",
            center={"lat": 50, "lon": 10},
            showcoastlines=True,
            coastlinecolor="Gray",
            showland=True,
            landcolor="lightgrey",
            showcountries=True
        )

        # Add hover template
        fig.update_traces(
            hovertemplate=(f"<b>%{{hovertext}}</b><br>Year: {selected_year}<br>Total Emission: %{{z}}")
        )
        fig.update_layout(
            coloraxis_colorbar=dict(
                title=f"Total Greenhouse Gas Emission in Year {selected_year}",
                tickvals=[0, 1e9, 2e9, 3e9, 4e9],  # Explicit tick so I can rename them to billion from b
                ticktext=["0 billion", "1 billion", "2 billion", "3 billion", "4 billion"],
            )
)
        return fig

    # Callback for the GHG aggregated trend graph in the second tab
    @app.callback(
        Output('ghg-aggregated-trend-graph', 'figure'),
        Input('ghg-aggregated-trend-graph', 'id')
    )
    def update_aggregated_ghg_graph(dummy_input):
        grouped_data = europe_result.groupby(['north_or_south', 'year'])['total_ghg'].sum().reset_index()
        fig2 = px.line(
            grouped_data,
            x='year',
            y='total_ghg',
            color='north_or_south',
            title="Total GHG Emissions Per North/South: Northern vs Southern Europe",
            labels={'year': 'Year', 'total_ghg': 'Total GHG Emissions'},
            color_discrete_map={'Northern': 'blue', 'Southern': 'red'}
        )
        fig2.update_traces(mode='lines', line=dict(width=2))
        fig2.update_layout(
            xaxis_title='Year',
            yaxis=dict(title='Total GHG Emissions', range=[0, 10e9]),
            legend_title='Region'
        )
        fig2.update_xaxes(range=[1990, 2000], tickmode='linear')

        # Add a shaded rectangle for 1995–1996, to make it stand out mroe
        fig2.add_shape(
            type="rect",
            x0=1995,
            x1=1996,
            y0=0,
            y1=10e9,
            fillcolor="rgba(128, 128, 128, 0.2)",
            line_width=0
        )
        fig2.add_vline(
            x=1995,
            line=dict(color="gray", dash="dash"),
            annotation_text="1995",
            annotation_position="top right"
        )
        return fig2

    # New callback for the aggregated temperature plot
    @app.callback(
        Output('temp-aggregated-1995-1997-graph', 'figure'),
        Input('temp-aggregated-1995-1997-graph', 'id')
    )

    def update_temp_aggregated(dummy_input):
        if not global_temps.empty:
            # Melt data into long format
            global_temps_long = global_temps.melt(
                id_vars=['Entity', 'Code', 'Year'],
                value_vars=['1995', '1996', '1997'],
                var_name='Year_Value',
                value_name='Temperature'
            )

            # Convert 'Year' to numeric for plotting
            global_temps_long['Year'] = pd.to_numeric(global_temps_long['Year'], errors='coerce')
            global_temps_long['Year_Value'] = pd.to_numeric(global_temps_long['Year_Value'], errors='coerce')

            # Aggregate temperatures by month and year
            aggregated_data = global_temps_long.groupby(['Year', 'Year_Value'])['Temperature'].mean().reset_index()

            # Generate figure
            fig = px.line(
                aggregated_data,
                x='Year',
                y='Temperature',
                color='Year_Value',
                title="Average Temperature (1995-1997) Across Selected Countries",
                labels={
                    'Year': 'Month',
                    'Temperature': 'Average Temperature (°C)',
                    'Year_Value': 'Year'
                },
                markers=True
            )

            # Explicitly map line colors for each year
            year_colors = {
                '1995': '#FFA500',  # Orange
                '1996': '#0000FF',  # Blue
                '1997': '#FF0000',  # Red
            }
            fig.for_each_trace(
                lambda trace: trace.update(
                    line=dict(color=year_colors[str(int(trace.name))], width=2),
                    marker=dict(color=year_colors[str(int(trace.name))])
                ) if trace.name in year_colors else None
            )

            # Map the "numeric" months to corresponding names.
            month_mapping = {
                1: 'January', 2: 'February', 3: 'March', 4: 'April',
                5: 'May', 6: 'June', 7: 'July', 8: 'August',
                9: 'September', 10: 'October', 11: 'November', 12: 'December'
            }
            fig.update_layout(
                xaxis=dict(
                    title='Month',
                    tickmode='array',
                    tickvals=list(month_mapping.keys()),
                    ticktext=list(month_mapping.values())
                ),
                yaxis=dict(title='Average Temperature (°C)'),
                legend_title_text='Year'
            )
            return fig
        else:
            return px.line(title="No Aggregated Temperature Data Available for 1995–1997")



    # Callback to update the markdown content dynamically
    @app.callback(
        Output('markdown-content', 'children'),
        Input('interval-component', 'n_intervals')
    )
    def update_markdown(n_intervals):
        # Reload the markdown
        return load_markdown(MARKDOWN_FILE_PATH)
