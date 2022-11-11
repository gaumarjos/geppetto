from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np

import geppetto

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

trace = geppetto.Geppetto(["tracks/The_missing_pass_W3_D2_.gpx"], debug=0, debug_plots=0, csv=1)
my_df = trace.df[0]

app.layout = html.Div([
    html.Table([
        html.Tbody([
            html.Tr([
                html.Td([
                    html.Div([
                        dcc.Graph(id='map_graph'),
                        dcc.Graph(id='elevation_graph'),
                        # dcc.Graph(id='hr_graph')
                    ],
                        style={'width': '100%',
                               'height': '100%',
                               'padding': '0 0',
                               'float': 'left'},
                        id="left_col"),
                ],
                    style={'width': '50%'}),
                html.Td([
                    html.Div([
                        html.Div([
                            dcc.Slider(0, 20, 2,
                                       value=4,
                                       id='power_filter_slider'
                                       ),
                            dcc.Graph(id='power_graph'),
                        ],
                            id="power"),
                        html.Div([
                            dcc.Dropdown(['100', '200', '500', '1000'],
                                         '500',
                                         id='gradient_resolution'),
                            dcc.Graph(id='gradient_graph')
                            # html.Pre(id='selected-data')
                        ],
                            id="debug"),
                    ],
                        style={'width': '100%',
                               'height': '100%',
                               'padding': '0 0',
                               'float': 'right'},
                        id="right_col"),
                ],
                    style={'width': '50%'})
            ])
        ]),
    ],
        style={"width": "100%"}),

])


@app.callback(
    Output('elevation_graph', 'figure'),
    Input('map_graph', 'hoverData'))
def update_elevation_graph(hoverData):
    title = "Elevation"
    hover_index = None
    if hoverData is not None:
        hover_index = hoverData['points'][0]['pointIndex']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=my_df['c_dist_geo2d'],
                             y=my_df['elev'],
                             mode='lines+markers',
                             name="",
                             line=dict(
                                 width=1,
                                 color="red"),
                             marker=dict(
                                 size=1,
                                 color="red")
                             ),
                  )
    if hover_index is not None:
        fig.add_trace(go.Scatter(x=[my_df.iloc[hover_index]['c_dist_geo2d']],
                                 y=[my_df.iloc[hover_index]['elev']],
                                 mode='markers',
                                 name="",
                                 marker=dict(
                                     size=16,
                                     color="yellow"),
                                 )
                      )

    fig.update_xaxes(showgrid=True,
                     showspikes=True,
                     title="2D distance (m)")
    fig.update_yaxes(showgrid=True)

    fig.add_annotation(x=0.02, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text="Elevation")
    fig.update_layout(height=225,
                      margin={'l': 10, 'b': 10, 'r': 10, 't': 10},
                      hovermode='x')
    return fig


@app.callback(
    Output('map_graph', 'figure'),
    Input('elevation_graph', 'selectedData'),
)
def update_map(selected_points):
    if selected_points is not None:
        selected_indexes = [d['pointIndex'] for d in selected_points['points']]
        fig = trace.plot_map(interval_unit="i",
                             interval=[min(selected_indexes), max(selected_indexes)] if len(selected_indexes) > 0 else [
                                 0, 0])
    else:
        fig = trace.plot_map(interval_unit="i",
                             interval=[0, 0])

    return fig


@app.callback(
    Output('gradient_graph', 'figure'),
    Input('elevation_graph', 'selectedData'),
    Input('gradient_resolution', 'value'),
)
def update_gradient(selected_points, gradient_resolution):
    if selected_points is not None:
        selected_indexes = [d['pointIndex'] for d in selected_points['points']]
        fig = trace.gradient(interval_unit="i",
                             interval=[min(selected_indexes), max(selected_indexes)] if len(selected_indexes) > 0 else [
                                 0, 0],
                             resolution=int(gradient_resolution),
                             show_map=True)
    else:
        fig = trace.gradient(interval_unit="i",
                             interval=[0, 0],
                             resolution=1000)

    return fig


@app.callback(
    Output('power_graph', 'figure'),
    Input('elevation_graph', 'selectedData'),
    Input('power_filter_slider', 'value'),
)
def update_power(selected_points, power_filter_slider):
    if selected_points is not None:
        selected_indexes = [d['pointIndex'] for d in selected_points['points']]
        fig = trace.estimate_power(interval_unit="i",
                                   interval=[min(selected_indexes), max(selected_indexes)] if len(
                                       selected_indexes) > 0 else [0, 0],
                                   filter_window=int(power_filter_slider))
    else:
        fig = trace.estimate_power(interval_unit="i",
                                   interval=[0, 0],
                                   filter_window=int(power_filter_slider))

    return fig


'''
@app.callback(
    Output('selected-data', 'children'),
    Input('elevation_graph', 'selectedData'))
def display_selected_data(selectedData):
    return json.dumps(selectedData, indent=2)
'''

if __name__ == '__main__':
    app.run_server(debug=True)
