from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import os
# import webbrowser

import mate

TRACK_DIRECTORY = "tracks/"

app = Dash(__name__,
           title='Geppetto',
           external_stylesheets=[dbc.themes.FLATLY],
           eager_loading=True,
           )

file_dropdown_list = mate.scan_files(TRACK_DIRECTORY)

app.layout = dbc.Container(
    children=[
        # Where the dataset is stored
        dcc.Store(id='store_df'),
        dcc.Store(id='store_df_moving'),

        # Top row: file selection bar
        dbc.Row(
            [
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                dcc.Dropdown(
                                    file_dropdown_list,
                                    file_dropdown_list[0]['value'],
                                    id='imported_files',
                                    searchable=True,
                                    clearable=False,
                                    style={
                                        "width": "100%",
                                    }
                                ),
                            ],
                        ),
                    ],
                    color="LightSlateGray",
                ),
            ]
        ),

        dbc.Row(
            [
                html.Div(
                    className="vspacer",
                )
            ]
        ),

        dbc.Row(
            [
                # Left column: map and stats on one row and elevation on another
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardHeader("Map", class_name="card_header"),
                                                dbc.CardBody(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Label(
                                                                    "Highlight value",
                                                                    style={
                                                                        "display": "inline-block",
                                                                        "width": "20%",
                                                                        "justifycontent": "center",
                                                                    }),
                                                                dcc.Dropdown(
                                                                    [
                                                                        {'label': 'Elevation', 'value': 'elev'},
                                                                        {'label': 'Speed', 'value': 'c_speed'},
                                                                        {'label': 'Distance',
                                                                         'value': 'c_dist_geo2d'}
                                                                    ],
                                                                    'elev',
                                                                    clearable=False,
                                                                    id='map_trace_color',
                                                                    style={
                                                                        "display": "inline-block",
                                                                        "width": "100%",
                                                                    },
                                                                ),
                                                            ],
                                                        ),
                                                        # Used by plotly bug workaround
                                                        dcc.Store(id='map_store'),
                                                        # Used to save the zoom info
                                                        dcc.Store(id='map_zoom_info'),
                                                        dcc.Graph(id='map_graph',
                                                                  animate=False,
                                                                  ),
                                                    ],
                                                ),
                                            ],
                                            color="Snow",
                                        ),
                                    ],
                                    width=8,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardHeader("Stats", class_name="card_header"),
                                                dbc.CardBody(
                                                    [
                                                        dcc.Markdown(style={
                                                            'textAlign': 'left',
                                                        },
                                                            id='markdown_stats'
                                                        ),
                                                    ],
                                                ),
                                            ],
                                            color="Snow",
                                        ),
                                    ],
                                    width=4,
                                ),
                            ]
                        ),

                        dbc.Row(
                            [
                                html.Div(
                                    className="vspacer",
                                )
                            ],
                        ),

                        dbc.Card(
                            [
                                dbc.CardHeader("Elevation and track selector", class_name="card_header"),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dcc.Graph(id='elevation_graph'),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                            color="Snow",
                        ),
                    ],
                    width=6,
                ),

                # Right column, all other plots in different rows
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Gradient", class_name="card_header"),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Row(
                                                            [
                                                                html.Div("Step (m)"),
                                                                dcc.Dropdown(['100', '200', '500', '1000'],
                                                                             '500',
                                                                             clearable=False,
                                                                             id='gradient_resolution'),
                                                                dcc.Checklist(
                                                                    id="slope_checkbox",
                                                                    options={
                                                                        'deg': 'Toggle % / deg',
                                                                    },
                                                                    value=[]
                                                                ),
                                                                dcc.Checklist(
                                                                    id="minimap_checkbox",
                                                                    options={
                                                                        'show': 'Show map',
                                                                    },
                                                                    value=[]
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                    width=2,
                                                ),

                                                dbc.Col(
                                                    [
                                                        dcc.Graph(id='gradient_graph')
                                                    ],
                                                    width=10,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                            color="Snow"
                        ),

                        dbc.Row(
                            [
                                html.Div(
                                    className="vspacer",
                                )
                            ]
                        ),

                        dbc.Card(
                            [
                                dbc.CardHeader("Power", class_name="card_header"),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                # Power settings
                                                dbc.Col(
                                                    [
                                                        html.Div("Filter width"),
                                                        dcc.Slider(0, 32, 2,
                                                                   value=4,
                                                                   id='power_filter_slider',
                                                                   vertical=True,
                                                                   marks={
                                                                       1: {'label': '1'},
                                                                       2: {'label': '2'},
                                                                       4: {'label': '4'},
                                                                       8: {'label': '8'},
                                                                       16: {'label': '16'},
                                                                       32: {'label': '32'}
                                                                   },
                                                                   ),
                                                    ],
                                                    width=1,
                                                ),

                                                # Power
                                                dbc.Col(
                                                    [
                                                        dbc.Row(
                                                            [
                                                                dcc.Graph(id='power_graph'),
                                                            ],
                                                        ),
                                                    ],
                                                    width=11,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                            color="Snow",
                        ),

                        # # Speed vs cadence
                        # dbc.Row(
                        #     [
                        #         # Spacer sx
                        #         dbc.Col(
                        #             [
                        #             ],
                        #             width=2
                        #         ),
                        #         # Gradient
                        #         dbc.Col(
                        #             [
                        #                 dbc.Row(
                        #                     [
                        #                         dcc.Graph(id='speed_cadence_curve_graph')
                        #                     ]
                        #                 ),
                        #             ],
                        #             width=10,
                        #         ),
                        #     ]
                        # ),
                    ],
                    width=6,
                ),
            ],
        ),
    ],
    fluid=True,
    style={"width": "100hw",
           # "height": "100vh",
           "background-color": "SlateGray",
           "font-size": "12"},
)


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
    Output('store_df', 'data'),
    Output('store_df_moving', 'data'),
    Input('imported_files', 'value')
)
def load_trace(filename):
    # Load trace
    if filename is not None:
        df_list, df_moving_list, _ = mate.load([TRACK_DIRECTORY + filename])
        df = df_list[0]
        df_moving = df_moving_list[0]
        return df.to_json(date_format='iso', orient='split'), df_moving.to_json(date_format='iso', orient='split')
    else:
        return None, None


# To keep the map zoom info up to date in "map_zoom_info"
@app.callback(
    Output('map_zoom_info', 'data'),
    Input('map_graph', 'relayoutData'),
    Input('map_zoom_info', 'data'),
)
def update_zoom_info(relayout_data, zoom_info):
    # print(relayout_data)
    if zoom_info is None:
        return relayout_data
    else:
        zoom_info.update(relayout_data)
        return zoom_info


@app.callback(
    # Output('map_graph', 'figure'),  # Replaced by the next line
    Output('map_store', 'data'),  # Used by plotly bug workaround
    Input('store_df', 'data'),
    Input('elevation_graph', 'selectedData'),
    Input('map_trace_color', 'value'),
    Input('elevation_graph', 'hoverData'),
    State('map_zoom_info', 'data')
)
def update_map(jsonified_df, selected_points, map_trace_color, hoverData, zoom):
    if jsonified_df is not None:
        df = pd.read_json(jsonified_df, orient='split')

        hover_index = None
        if hoverData is not None:
            hover_index = hoverData['points'][0]['pointIndex']

        if selected_points is not None:
            selected_indexes = [d['pointIndex'] for d in selected_points['points']]
            fig = mate.plot_map(df=df,
                                map_trace_color_param=map_trace_color,
                                interval_unit="i",
                                interval=[min(selected_indexes), max(selected_indexes)] if len(
                                    selected_indexes) > 0 else [0, 0],
                                hover_index=hover_index,
                                zoom=zoom)
        else:
            fig = mate.plot_map(df=df,
                                map_trace_color_param=map_trace_color,
                                interval_unit="i",
                                interval=[0, 0],
                                hover_index=hover_index,
                                zoom=zoom)

        return fig


# Used by plotly bug workaround: change the id in the state and output to change the id of your plot
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
    Output('map_graph', 'figure'),
    Input('map_store', 'data'),
    State('map_graph', 'id')
)


@app.callback(
    Output('elevation_graph', 'figure'),
    Input('store_df', 'data'),
    # Input('map_graph', 'hoverData')
)
def update_elevation_graph(jsonified_df):  # , hoverData):
    if jsonified_df is not None:
        df = pd.read_json(jsonified_df, orient='split')
        hover_index = None
        # if hoverData is not None:
        #     hover_index = hoverData['points'][0]['pointIndex']

        fig = mate.plot_elevation(df=df,
                                  hover_index=hover_index)

        return fig


@app.callback(
    Output('gradient_graph', 'figure'),
    Input('store_df', 'data'),
    Input('elevation_graph', 'selectedData'),
    Input('gradient_resolution', 'value'),
    Input('slope_checkbox', 'value'),
    Input('minimap_checkbox', 'value'),
)
def update_gradient(jsonified_df, selected_points, gradient_resolution, slope_checkbox, minimap_checkbox):
    if jsonified_df is not None:
        df = pd.read_json(jsonified_df, orient='split')
        if selected_points is not None:
            selected_indexes = [d['pointIndex'] for d in selected_points['points']]
            fig = mate.gradient(df=df,
                                interval_unit="i",
                                interval=[min(selected_indexes), max(selected_indexes)] if len(
                                    selected_indexes) > 0 else [0, 0],
                                resolution=int(gradient_resolution) if gradient_resolution is not None else 1000,
                                slope_unit="deg" if "deg" in slope_checkbox else "per",
                                show_map=True if "show" in minimap_checkbox else False)
        else:
            fig = mate.gradient(df=df,
                                interval_unit="i",
                                interval=[0, 0],
                                resolution=1000,
                                slope_unit="deg" if "deg" in slope_checkbox else "per")

        return fig


@app.callback(
    Output('power_graph', 'figure'),
    Input('store_df', 'data'),
    Input('store_df_moving', 'data'),
    Input('elevation_graph', 'selectedData'),
    Input('power_filter_slider', 'value'),
)
def update_power(jsonified_df, jsonified_df_moving, selected_points, power_filter_slider):
    if jsonified_df is not None:
        df = pd.read_json(jsonified_df, orient='split')
        df_moving = pd.read_json(jsonified_df_moving, orient='split')
        if selected_points is not None:
            selected_indexes = [d['pointIndex'] for d in selected_points['points']]
            fig = mate.estimate_power(df=df,
                                      df_moving=df_moving,
                                      interval_unit="i",
                                      interval=[min(selected_indexes), max(selected_indexes)] if len(
                                          selected_indexes) > 0 else [0, 0],
                                      filter_window=int(
                                          power_filter_slider) if power_filter_slider is not None else 0)
        else:
            fig = mate.estimate_power(df=df,
                                      df_moving=df_moving,
                                      interval_unit="i",
                                      interval=[0, 0],
                                      filter_window=int(
                                          power_filter_slider) if power_filter_slider is not None else 0)

        return fig


# @app.callback(
#     Output('speed_cadence_curve_graph', 'figure'),
#     Input('store_df', 'data'),
#     Input('store_df_moving', 'data'),
#     Input('elevation_graph', 'selectedData'),
# )
# def update_speed_cadence_curve(jsonified_df, jsonified_df_moving, selected_points):
#     if jsonified_df is not None:
#         df = pd.read_json(jsonified_df, orient='split')
#         df_moving = pd.read_json(jsonified_df_moving, orient='split')
#         if selected_points is not None:
#             selected_indexes = [d['pointIndex'] for d in selected_points['points']]
#             fig = mate.cadence_speed_curve(df=df,
#                                                df_moving=df_moving,
#                                                interval_unit="i",
#                                                interval=[min(selected_indexes), max(selected_indexes)] if len(
#                                                    selected_indexes) > 0 else [0, 0])
#         else:
#             fig = mate.cadence_speed_curve(df=df,
#                                                df_moving=df_moving,
#                                                interval_unit="i",
#                                                interval=[0, 0])
#
#         return fig


@app.callback(
    Output('markdown_stats', 'children'),
    Input('store_df', 'data'),
    Input('store_df_moving', 'data'),
    Input('elevation_graph', 'selectedData'),
)
def update_stats(jsonified_df, jsonified_df_moving, elevation_graph):
    if (jsonified_df is not None) and (jsonified_df_moving is not None):
        df = pd.read_json(jsonified_df, orient='split')
        df_moving = pd.read_json(jsonified_df_moving, orient='split')
        return mate.stats(df=df, df_moving=df_moving)


'''
@app.callback(
    Output('selected-data', 'children'),
    Input('elevation_graph', 'selectedData'))
def display_selected_data(selectedData):
    return json.dumps(selectedData, indent=2)
'''

if __name__ == '__main__':
    # webbrowser.open_new_tab("http://localhost:8050/")
    app.run_server(debug=True)
