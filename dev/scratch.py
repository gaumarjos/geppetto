# import app
#
# print(app.get_file_list())
#import mate
#import geppetto


print(-30 % 120)
print(124 % 120)


import pandas as pd
import numpy as np

# create a list of 120 numbers ranging from 0 to 119
c_dist_geo2d_values = list(range(120))

# create a dataframe with one column named "c_dist_geo2d"
df = pd.DataFrame({'c_dist_geo2d': c_dist_geo2d_values})

# print the dataframe
print(df)

print(df.iloc[-1:10])


a = np.arange(0, 120)
print(a)
print(a[-30:110])




#geppetto.scan_files(app.TRACK_DIRECTORY, verbose=True)
"""
from geopy.geocoders import Nominatim
import gpxpy

import geppetto


def location_info(file):
        gpx = gpxpy.parse(open(file, 'r'))
        lon = gpx.tracks[0].segments[0].points[0].longitude
        lat = gpx.tracks[0].segments[0].points[0].latitude

        geolocator = Nominatim(user_agent="geoapiExercises")
        location = geolocator.reverse(str(lat) + "," + str(lon))

        return location.raw["address"]
        # return location.raw["display_name"]

# {'road': 'Strada Provinciale per Succiso', 'suburb': 'Pieve San Vincenzo', 'village': 'Cecciola', 'municipality': 'Ventasso', 'county': "Reggio nell'Emilia", 'ISO3166-2-lvl6': 'IT-RE', 'state': 'Emilia-Romagna', 'ISO3166-2-lvl4': 'IT-45', 'country': 'Italia', 'country_code': 'it'}
print(geppetto.location_info("tracks/The_local_4_or_5_passes.gpx"))


"""



# import datetime
# str = "2022-09-01T08:04:05.000Z"
# date = datetime.datetime.strptime(str, "%Y-%m-%dT%H:%M:%S.%fZ")
#





'''
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, State


app = Dash(__name__,
           title='Geppetto',
           external_stylesheets=[dbc.themes.LUX],
           )


app.layout = html.Div(
    [
        dbc.Row(dbc.Col(html.Div("A single, half-width column"), width=6)),
        dbc.Row(
            dbc.Col(html.Div("An automatically sized column"), width="auto")
        ),
        dbc.Row(
            [
                dbc.Col(html.Div("One of three columns"), width=3),
                dbc.Col(html.Div("One of three columns")),
                dbc.Col(html.Div("One of three columns"), width=3),
            ]
        ),
    ]
)



if __name__ == '__main__':
    app.run_server(debug=True)

'''

'''
def update_map3(selected_points):
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

'''



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