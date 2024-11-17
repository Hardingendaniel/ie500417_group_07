from dash import html, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px
from data.data import data

# List of continents
continents = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']

# The ISO codes of European countries
europe = ['NOR', 'SWE', 'FIN', 'DNK', 'ISL', 'GBR', 'IRL', 'DEU', 'NLD', 'POL',
          'EST', 'LVA', 'LTU', 'RUS', 'BLR', 'CZE', 'SVK', 'FRA', 'UKR',
          'CHE', 'AUT', 'HUN', 'BEL', 'LUX', 'MDA', 'ITA', 'ESP', 'PRT', 'GRC', 'MLT',
          'CYP', 'HRV', 'SVN', 'BGR', 'ROU', 'ALB', 'BIH', 'MNE', 'MKD', 'MCO', 'SMR',
          'VAT', 'AND', 'SRB']

# Filter and aggregate data for continents
continent_data = data[data['country'].isin(continents)][['year', 'country', 'co2']]
continent_aggregated = (
    continent_data.groupby(['year', 'country'])
    .sum()
    .reset_index()
    .rename(columns={'country': 'region'})
)

# Combine all data
combined_data = pd.concat([continent_aggregated])

# Filter data for the year 1850 and later
combined_data = combined_data[combined_data['year'] >= 1850]

# Create the figure for continents
fig_continents = px.line(
    combined_data,
    x='year',
    y='co2',
    color='region',
    title='CO₂ Emissions Over Time: By Region',
    labels={'co2': 'Total CO₂ Emissions (Metric Tons)', 'year': 'Year', 'region': 'Region'}
)

# Filter and aggregate data for European countries
europe_data = data[data['iso_code'].isin(europe)][['year', 'iso_code', 'country', 'co2']]
europe_aggregated = (
    europe_data.groupby(['year', 'country'])
    .sum()
    .reset_index()
)

europe_aggregated = europe_aggregated[europe_aggregated['year'] >= 1850]

# Layout for the page
layout = html.Div([
    html.H1('CO₂ Emissions Over Time: By Region'),
    dcc.Graph(figure=fig_continents),
    html.H1('CO₂ Emissions Over Time: By European Countries'),

    # Dropdown for selecting European countries
    dcc.Dropdown(
        id='europe-country-selector',
        options=[{'label': name, 'value': name} for name in europe_aggregated['country'].unique()],
        value=['Norway', 'Sweden', 'Germany', 'United Kingdom'],  # Default selection
        multi=True,
        placeholder='Select countries to display',
        style={'margin-bottom': '20px'}
    ),

    # Graph for European countries
    dcc.Graph(id='europe-graph')
])


# Callback to update the European countries graph based on selection
@callback(
    Output('europe-graph', 'figure'),
    Input('europe-country-selector', 'value')
)
def update_europe_graph(selected_countries):
    # Filter data based on selected countries
    filtered_data = europe_aggregated[europe_aggregated['country'].isin(selected_countries)]

    # Create the updated figure
    fig = px.line(
        filtered_data,
        x='year',
        y='co2',
        color='country',
        title='CO₂ Emissions Over Time: By European Countries',
        labels={'co2': 'Total CO₂ Emissions (Metric Tons)', 'year': 'Year', 'country': 'Country'}
    )
    return fig



