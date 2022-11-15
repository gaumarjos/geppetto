import dash
from dash import Input, Output, html, dcc, callback, ctx, State
import plotly.express as px
import plotly.graph_objects as go

import plotly.express as px
import pandas as pd

# IMPORTANT: set eager_loading=True
app = dash.Dash(__name__, eager_loading=True)

us_cities = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/datasets/master/us-cities-top-1k.csv"
)


def us_fig(data):
    fig = px.scatter_mapbox(
        data,
        lat="lat",
        lon="lon",
        hover_name="City",
        hover_data=["State", "Population"],
        color_discrete_sequence=["fuchsia"],
        zoom=3,
        height=300,
    )
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(mapbox_bounds={"west": -180, "east": -50, "south": 20, "north": 90})

    return fig


app.layout = html.Div([
    dcc.Dropdown(id='states', options=list(us_cities.State.unique()), value=['California'], multi=True),
    dcc.Store(id='fig_store'),
    dcc.Graph(id='fig1')
])


@callback(
    Output('fig_store', 'data'),
    Input('states', 'value')
)
def update_plot(states):
    data = us_cities[us_cities['State'].isin(states)]
    new_fig1 = us_fig(data)

    return new_fig1


# change the id in the state and output to change the id of your plot
app.clientside_callback(
    '''
    function (figure, graph_id) {
        if(figure === undefined) {
            return {'data': [], 'layout': {}};
        }
        var graphDiv = document.getElementById(graph_id);
        var data = figure.data;
        var layout = figure.layout;        
        Plotly.newPlot(graphDiv, data, layout);
    }
    ''',
    Output('fig1', 'figure'),
    Input('fig_store', 'data'),
    State('fig1', 'id')
)

if __name__ == '__main__':
    app.run_server(debug=True)