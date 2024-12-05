import pandas as pd
import plotly.express as px
from dash import dcc, html


class AnalyticalOperations:
    def __init__(self, data, europe):
        self.data = data
        self.europe = europe

    def compute_analytical_factors(self, selected_year):
        filtered_data2 = self.data[(self.data['year'] == selected_year) & (self.data['iso_code'].notna())].copy()
        highest_total_ghg_anal = filtered_data2.nlargest(5, 'total_ghg')[['country', 'total_ghg']]
        highest_total_ghg_anal = highest_total_ghg_anal.merge(
            filtered_data2[['country', 'population', 'gdp']],
            on='country',
        )
        filtered_data2['pop_rank'] = filtered_data2['population'].rank(method='min', ascending=False)
        filtered_data2['gdp_rank'] = filtered_data2['gdp'].rank(method='min', ascending=False)

        highest_total_ghg_anal['Population_with_rank'] = highest_total_ghg_anal.apply(
            lambda
                row: f"{row['population']:,} ({int(filtered_data2[filtered_data2['country'] == row['country']]['pop_rank'].values[0])})",
            axis=1
        )
        highest_total_ghg_anal['GDP_with_rank'] = highest_total_ghg_anal.apply(
            lambda row: int(filtered_data2[filtered_data2['country'] == row['country']]['gdp_rank'].values[0]),
            axis=1
        )
        rest_of_world_ghg = filtered_data2['total_ghg'].sum() - highest_total_ghg_anal['total_ghg'].sum()
        global_pie_data = highest_total_ghg_anal[['country', 'total_ghg']].copy()
        top_5_world_ghg = highest_total_ghg_anal[['country', 'total_ghg', 'population']].copy()
        rest_of_world_population = filtered_data2['population'].sum() - top_5_world_ghg['population'].sum()
        world_pie_pop_data = pd.concat([
            top_5_world_ghg[['country', 'population']],
            pd.DataFrame({'country': ['Rest of World'], 'population': [rest_of_world_population]})
        ], ignore_index=True)
        global_pie_data = pd.concat([
            global_pie_data,
            pd.DataFrame({'country': ['Rest of the World'], 'total_ghg': [rest_of_world_ghg]})
        ], ignore_index=True)

        europe_data = filtered_data2[filtered_data2['iso_code'].isin(self.europe)].copy()
        europe_data['pop_rank'] = europe_data['population'].rank(method='min', ascending=False)
        europe_data['gdp_rank'] = europe_data['gdp'].rank(method='min', ascending=False)
        highest_europe_total_ghg_anal = europe_data.nlargest(5, 'total_ghg')[['country', 'total_ghg']]
        highest_europe_total_ghg_anal = highest_europe_total_ghg_anal.merge(
            europe_data[['country', 'population', 'gdp']],
            on='country',
        )
        highest_europe_total_ghg_anal['Population_with_rank'] = highest_europe_total_ghg_anal.apply(
            lambda
                row: f"{row['population']:,} ({int(europe_data[europe_data['country'] == row['country']]['pop_rank'].values[0])})",
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
            pd.DataFrame({'country': ['Rest of Europe'], 'population': [rest_of_europe_population]})
        ], ignore_index=True)

        fig = px.bar(
            highest_total_ghg_anal,
            x='country',
            y='total_ghg',
            hover_data={'total_ghg': False, 'Population_with_rank': True, 'GDP_with_rank': True},
            labels={'total_ghg': 'Total GHG Emissions', 'country': 'Country'},
            title='Top 5 Countries by Total GHG Emissions'
        )
        fig.update_yaxes(range=[0, None], fixedrange=True)

        fig.update_traces(hovertemplate='<br>'.join([
            'Country: %{x}',
            'Total GHG Emissions: %{y}',
            'Population (Rank): %{customdata[0]}',
            'GDP Rank: %{customdata[1]}'
        ]))

        bar_chart = dcc.Graph(
            figure=fig
        )

        fig_europe = px.bar(
            highest_europe_total_ghg_anal,
            x='country',
            y='total_ghg',
            hover_data={'total_ghg': False, 'Population_with_rank': True, 'GDP_with_rank': True},
            labels={'total_ghg': 'Total GHG Emissions', 'country': 'Country'},
            title='Top 5 Countries by Total GHG Emissions in Europe'
        )
        fig_europe.update_yaxes(range=[0, None], fixedrange=True)

        fig.update_traces(hovertemplate='<br>'.join([
            'Country: %{x}',
            'Total GHG Emissions: %{y}',
            'Population (Rank): %{customdata[0]}',
            'GDP Rank: %{customdata[1]}'
        ]))

        bar_chart_europe = dcc.Graph(
            figure=fig_europe
        )

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
            title='Population Distribution (Top 5 vs Rest of World)'
        )
        world_pop_chart = dcc.Graph(figure=world_pop_fig)

        world_section = html.Div(
            style={'display': 'flex', 'justifyContent': 'space-around', 'align-items': 'center',
                   'margin-bottom': '20px'},
            children=[
                html.Div(global_pie_chart, style={'flex': '1', 'padding': '10px'}),
                html.Div(world_pop_chart, style={'flex': '1', 'padding': '10px'})
            ]
        )

        europe_pie_fig = px.pie(
            europe_pie_data,
            names='country',
            values='total_ghg',
            title='GHG Emissions Distribution (Top 5 European vs Rest of Europe)'
        )

        europe_pie_chart = dcc.Graph(
            figure=europe_pie_fig
        )

        europe_pop_fig = px.pie(
            europe_pie_pop_data,
            names='country',
            values='population',
            title='Respective countries Population Distribution (Top 5 vs Rest of Europe)'
        )
        europe_pop_chart = dcc.Graph(figure=europe_pop_fig)

        europe_section = html.Div(
            style={'display': 'flex', 'justifyContent': 'space-around', 'align-items': 'center',
                   'margin-bottom': '20px'},
            children=[
                html.Div(europe_pie_chart, style={'flex': '1', 'padding': '10px'}),
                html.Div(europe_pop_chart, style={'flex': '1', 'padding': '10px'})
            ]
        )

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
