from dash import html, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px
from data.data import data

europe = ['NOR', 'SWE', 'FIN', 'DNK', 'ISL', 'GBR', 'IRL', 'DEU', 'NLD', 'POL',
          'EST', 'LVA', 'LTU', 'RUS', 'BLR', 'CZE', 'SVK', 'FRA', 'UKR',
          'CHE', 'AUT', 'HUN', 'BEL', 'LUX', 'MDA',  'ITA', 'ESP', 'PRT', 'GRC', 'MLT',
          'CYP', 'HRV', 'SVN', 'BGR', 'ROU', 'ALB', 'BIH', 'MNE', 'MKD', 'MCO', 'SMR',
          'VAT', 'AND', 'SRB']

european_data = data[data['iso_code'].isin(europe)][['year', 'co2']]
rest_of_world_data = data[~data['iso_code'].isin(europe)][['year', 'co2']]



# Aggregate CO₂ emissions by year for both Europe and the rest of the world
europe_aggregated = european_data.groupby('year').sum().reset_index()
europe_aggregated['region'] = 'Europe'  # Add region column

rest_of_world_aggregated = rest_of_world_data.groupby('year').sum().reset_index()
rest_of_world_aggregated = rest_of_world_aggregated.assign(region='Rest of the World')

# Combine the data for plotting
combined_data = pd.concat([europe_aggregated, rest_of_world_aggregated])

combined_data = combined_data[combined_data['year'] >= 1850]

# Create the figure with both Europe and Rest of the World data
fig = px.line(
    combined_data,
    x='year',
    y='co2',
    color='region',  # This will add separate lines for "Europe" and "Rest of the World"
    title='CO₂ Emissions Over Time: Europe vs. Rest of the World',
    labels={'co2': 'Total CO₂ Emissions (Metric Tons)', 'year': 'Year', 'region': 'Region'}
)

# Layout for the page
layout = html.Div([
    html.H1('Comparison of CO₂ Emissions: Europe vs. Rest of the World'),
    dcc.Graph(figure=fig)
])