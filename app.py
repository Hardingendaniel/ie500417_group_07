# app.py
from dash import Dash, html, dcc
import pages.analyse as analyse
import pages.prediction as prediction
import pages.visualization1 as visualization1
import pages.visualization2 as visualization2

# Initialize the app
app = Dash(__name__)
visualization2.init_callbacks(app)

# App layout with tabs
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Analyse', children=analyse.layout),
        dcc.Tab(label='Prediction', children=prediction.layout),
        dcc.Tab(label='Visualization', children=visualization1.layout),
        dcc.Tab(label='Visualization2', children=visualization2.layout)
    ])
])

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
