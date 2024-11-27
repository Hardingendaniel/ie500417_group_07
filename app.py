# app.py
from dash import Dash, html, dcc
import pages.analyse as analyse
import pages.prediction as prediction
import pages.visualization1 as visualization1
import pages.visualizationOfEurope as visualizationOfEurope

# Initialize the app
app = Dash(__name__)
visualizationOfEurope.init_callbacks(app)
analyse.init_callbacksalso(app)


# App layout with tabs
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(className='tabs-title',label='Analyse', children=analyse.layout),
        dcc.Tab(className='tabs-title',label='Prediction', children=prediction.layout),
        dcc.Tab(className='tabs-title',label='Visualization', children=visualization1.layout),
        dcc.Tab(className='tabs-title',label='Visualization of Europe and Arctic Winter', children=visualizationOfEurope.layout)
    ])
])

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
