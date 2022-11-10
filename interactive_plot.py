from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')
mydf = pd.read_csv('tracks/The_missing_pass_W3_D2_.csv')

app.layout = html.Div([
    html.Table([
        html.Thead([
            html.Tr([
                html.Th([
                    "Geppetto"
                ])
            ])
        ]),
        html.Tbody([
            html.Tr([
                html.Td([
                    html.Div([
                        dcc.Graph(id='map_graph'),
                        dcc.Graph(id='elevation_graph'),
                        dcc.Graph(id='hr_graph')
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
                            html.Button('Analyze', id='btn-analyze', n_clicks=0)
                        ],
                            id="buttons"),
                        html.Div([
                            dcc.Graph(id='power_graph'),
                            dcc.Graph(id='gradient_graph')
                        ],
                            id="analyses"),
                        html.Div([
                            html.Pre(id='selected-data')],
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
            fig = px.scatter(x=mydf.iloc[selected_indexes]['lon'],
                             y=mydf.iloc[selected_indexes]['lat'],
                             # x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
                             # y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
                             # hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name']
                             )
        else:
            fig = px.scatter(x=mydf['lon'],
                             y=mydf['lat'],
                             # x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
                             # y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
                             # hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name']
                             )
    else:
        if len(selected_indexes) > 0:
            print("should be partial")
            print(selected_indexes)
            data = [go.Scattermapbox(lat=mydf.iloc[selected_indexes]['lat'],
                                     lon=mydf.iloc[selected_indexes]['lon'],
                                     mode='markers',
                                     marker=dict(
                                         size=8,
                                         color='yellow',
                                         opacity=1.0)
                                     )]
        else:
            data = [go.Scattermapbox(lat=mydf['lat'],
                                     lon=mydf['lon'],
                                     mode='lines',
                                     line=dict(
                                         width=2,
                                         color='red')
                                     )]

        layout = go.Layout(autosize=False,
                           mapbox=dict(bearing=0,
                                       pitch=0,
                                       zoom=10,
                                       center=go.layout.mapbox.Center(
                                           lon=np.mean(mydf["lon"]),
                                           lat=np.mean(mydf["lat"])
                                       ),
                                       # accesstoken=map_token,
                                       style="open-street-map"),
                           margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
                           hovermode='closest'
                           )

        fig = go.Figure(data=data, layout=layout)

    # fig.update_traces(customdata=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'])

    # fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')

    # fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')

    # fig.update_layout()

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
    fig.add_trace(go.Scatter(x=mydf['c_dist_geo2d'],
                             y=mydf['elev'],
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
        fig.add_trace(go.Scatter(x=[mydf.iloc[hover_index]['c_dist_geo2d']],
                                 y=[mydf.iloc[hover_index]['elev']],
                                 mode='markers',
                                 name="",
                                 marker=dict(
                                     size=16,
                                     color="yellow"),
                                 )
                      )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text="Elevation")
    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
    return fig


@app.callback(
    Output('hr_graph', 'figure'),
    Input('map_graph', 'hoverData'))
def update_hr_graph(hoverData):
    title = "HR"
    fig = px.scatter(mydf, x='time', y='hr')
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


if __name__ == '__main__':
    app.run_server(debug=True)
