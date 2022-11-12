from dash import Dash, html, dcc, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
import base64
import datetime
import io
import os

import geppetto

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

UPLOAD_DIRECTORY = "uploaded_files"

app = Dash(__name__,
           title='Geppetto',
           external_stylesheets=external_stylesheets)

trace = geppetto.Geppetto(["tracks/The_missing_pass_W3_D2_.gpx"], debug=0, debug_plots=0, csv=1)

markdown_text = '''
stats go here'''

app.layout = html.Div([
    # Where the dataset is stored
    dcc.Store(id='store_dataframe'),

    html.Table([
        html.Tbody([
            html.Tr([
                # dcc.Upload(
                #     id='upload-data',
                #     children=html.Div([
                #         'Drag and Drop or ',
                #         html.A('Select File')
                #     ]),
                #     style={
                #         'width': '100%',
                #         'height': '40px',
                #         'lineHeight': '40px',
                #         'borderWidth': '1px',
                #         'borderStyle': 'dashed',
                #         'borderRadius': '5px',
                #         'textAlign': 'center',
                #         'margin': '10px'
                #     },
                #     # Allow multiple files to be uploaded
                #     multiple=False
                # ),
                # html.Div(id='output-data-upload'),
                dcc.Dropdown(os.listdir("tracks/"),
                             os.listdir("tracks/")[0],
                             id='imported_files'),
            ]),
            html.Tr([
                html.Td([
                    html.Div([
                        dcc.Dropdown([{'label': 'Elevation', 'value': 'elev'},
                                      {'label': 'Speed', 'value': 'c_speed'},
                                      {'label': 'Distance', 'value': 'c_dist_geo2d'}],
                                     'elev',
                                     id='map_trace_color'),
                        dcc.Graph(id='map_graph'),
                        dcc.Graph(id='elevation_graph'),
                        dcc.Markdown(children=markdown_text, id='markdown_stats')
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


# def save_file(name, content):
#     """Decode and store a file uploaded with Plotly Dash."""
#     data = content.encode("utf8").split(b";base64,")[1]
#     with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
#         fp.write(base64.decodebytes(data))
#
#
# def uploaded_files():
#     """List the files in the upload directory."""
#     files = []
#     for filename in os.listdir(UPLOAD_DIRECTORY):
#         path = os.path.join(UPLOAD_DIRECTORY, filename)
#         if os.path.isfile(path):
#             files.append(filename)
#     return files
#
#
# @app.callback(
#     Output('output-data-upload', 'children'),
#     Input("upload-data", "filename"),
#     Input("upload-data", "contents"),
# )
# def update_output(uploaded_filename, uploaded_file_content):
#     """Save uploaded files and regenerate the file list."""
#
#     if uploaded_filename is not None and uploaded_file_content is not None:
#         save_file(uploaded_filename, uploaded_file_content)
#         print(uploaded_filename)
#
#     files = uploaded_files()
#     if len(files) == 0:
#         return [html.Li("No files yet!")]
#     else:
#         return files
#         # return [html.Li(file_download_link(filename)) for filename in files]


@app.callback(
    Output('store_dataframe', 'data'),
    Input('imported_files', 'value')
)
def load_trace(filename):
    # Load trace
    trace = geppetto.Geppetto(["tracks/" + filename])
    return trace


@app.callback(
    Output('elevation_graph', 'figure'),
    Input('map_graph', 'hoverData'))
def update_elevation_graph(hoverData):
    hover_index = None
    if hoverData is not None:
        hover_index = hoverData['points'][0]['pointIndex']

    fig = trace.plot_elevation(hover_index=hover_index)

    return fig


@app.callback(
    Output('map_graph', 'figure'),
    Input('elevation_graph', 'selectedData'),
    Input('map_trace_color', 'value')
)
def update_map(selected_points, map_trace_color):
    if selected_points is not None:
        selected_indexes = [d['pointIndex'] for d in selected_points['points']]
        fig = trace.plot_map(map_trace_color_param=map_trace_color,
                             interval_unit="i",
                             interval=[min(selected_indexes), max(selected_indexes)] if len(selected_indexes) > 0 else [
                                 0, 0])
    else:
        fig = trace.plot_map(map_trace_color_param=map_trace_color,
                             interval_unit="i",
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


@app.callback(
    Output('markdown_stats', 'children'),
    Input('elevation_graph', 'selectedData'),
)
def update_stats(elevation_graph):
    return trace.stats()


'''
@app.callback(
    Output('selected-data', 'children'),
    Input('elevation_graph', 'selectedData'))
def display_selected_data(selectedData):
    return json.dumps(selectedData, indent=2)
'''

if __name__ == '__main__':
    app.run_server(debug=True)
