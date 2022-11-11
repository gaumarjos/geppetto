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
                            dcc.Graph(id='power_graph'),
                            dcc.Slider(100,
                                       1000,
                                       100,
                                       id='crossfilter-year--slider',
                                       value=500,
                                       ),
                            dcc.Graph(id='gradient_graph')
                        ],
                            id="analyses"),
                        html.Div([
                            # html.Pre(id='selected-data')
                        ],
                            id="debug")
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
    Output('map_graph', 'figure'),
    Input('elevation_graph', 'selectedData'),
)
def update_map(selected_points):
    selected_indexes = []
    if selected_points is not None:
        selected_indexes = [d['pointIndex'] for d in selected_points['points']]

    if 0:
        if len(selected_indexes) > 0:
            fig = px.scatter(x=my_df.iloc[selected_indexes]['lon'],
                             y=my_df.iloc[selected_indexes]['lat'],
                             # x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
                             # y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
                             # hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name']
                             )
        else:
            fig = px.scatter(x=my_df['lon'],
                             y=my_df['lat'],
                             # x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
                             # y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
                             # hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name']
                             )
    else:
        if len(selected_indexes) > 0:
            print("should be partial")
            print(selected_indexes)
            data = [go.Scattermapbox(lat=my_df.iloc[selected_indexes]['lat'],
                                     lon=my_df.iloc[selected_indexes]['lon'],
                                     mode='markers',
                                     marker=dict(
                                         size=8,
                                         color='yellow',
                                         opacity=1.0)
                                     )]
        else:
            data = [go.Scattermapbox(lat=my_df['lat'],
                                     lon=my_df['lon'],
                                     mode='lines',
                                     line=dict(
                                         width=2,
                                         color='red'),
                                     hovertext=my_df['c_dist_geo2d'],
                                     )]

        layout = go.Layout(autosize=False,
                           mapbox=dict(bearing=0,
                                       pitch=0,
                                       zoom=10,
                                       center=go.layout.mapbox.Center(
                                           lon=np.mean(my_df["lon"]),
                                           lat=np.mean(my_df["lat"])
                                       ),
                                       # accesstoken=map_token,
                                       style="open-street-map"),
                           margin={'l': 10, 'b': 10, 't': 10, 'r': 10},
                           hovermode='closest'
                           )

        fig = go.Figure(data=data, layout=layout)

    # fig.update_traces(customdata=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'])
    # fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')
    # fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')

    return fig


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
    Output('gradient_graph', 'figure'),
    Input('elevation_graph', 'selectedData'),
)
def update_gradient(selected_points):
    if selected_points is not None:
        selected_indexes = [d['pointIndex'] for d in selected_points['points']]
        fig = trace.gradient(interval_unit="i",
                             interval=[min(selected_indexes), max(selected_indexes)],
                             show_map=True)
    else:
        fig = trace.gradient(interval_unit="i",
                             interval=[0, 0])

    return fig


@app.callback(
    Output('power_graph', 'figure'),
    Input('elevation_graph', 'selectedData'),
)
def update_power(selected_points):
    if selected_points is not None:
        selected_indexes = [d['pointIndex'] for d in selected_points['points']]
        fig = trace.estimate_power(interval_unit="i", interval=[min(selected_indexes), max(selected_indexes)])
    else:
        fig = trace.estimate_power(interval_unit="i", interval=[0, 0])

    return fig


'''
@app.callback(
    Output('hr_graph', 'figure'),
    Input('map_graph', 'hoverData'))
def update_hr_graph(hoverData):
    title = "HR"
    fig = px.scatter(my_df, x='time', y='hr')
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text="HR")
    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
    return fig

@app.callback(
    Output('selected-data', 'children'),
    Input('elevation_graph', 'selectedData'))
def display_selected_data(selectedData):
    return json.dumps(selectedData, indent=2)
'''

if __name__ == '__main__':
    app.run_server(debug=True)
