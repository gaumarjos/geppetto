# https://thatmaceguy.github.io/python/gps-data-analysis-intro/
# https://plotly.com/python/mapbox-layers/


import numpy as np
import gpxpy
import gpxpy.gpx
import plotly.express as px
import pandas as pd
from geopy import distance
from math import sqrt, floor
import datetime
import plotly.graph_objects as go


class Geppetto():

    def __init__(self, gpx_file):
        gpx = gpxpy.parse(open(gpx_file, 'r'))

        points = gpx.tracks[0].segments[0].points
        self.df = pd.DataFrame(columns=['lon', 'lat', 'elev', 'time'])
        for point in points:
            self.df = pd.concat([self.df, pd.DataFrame(
                data={'lon': point.longitude, 'lat': point.latitude, 'elev': point.elevation, 'time': point.time},
                index=[0])],
                                axis=0, join='outer', ignore_index=True)

        # print(self.df)
        # print(self.df.describe())

        if 0:
            fig_1 = px.scatter(self.df, x='lon', y='lat', template='plotly_dark')
            fig_1.show()

        if 0:
            fig_3 = px.scatter_3d(self.df, x='lon', y='lat', z='elev', color='elev', template='plotly_dark')
            fig_3.update_traces(marker=dict(size=2), selector=dict(mode='markers'))
            fig_3.show()

        if 0:
            fig_4 = px.line_mapbox(self.df, lat='lat', lon='lon', hover_name='time', mapbox_style="open-street-map",
                                   zoom=11)
            fig_4.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
            fig_4.show()

        # first create lists to store the results, these will be appended ot the dataframe at the end
        # Note: i'll be working from the gps_points object directly and then appending results into the dataframe. It would make a lot more sense to operate directly on the dataframe.

        delta_elev = [0]  # change in elevation between records
        delta_time = [0]  # time interval between records
        delta_sph2d = [0]  # segment distance from spherical geometry only
        delta_sph3d = [0]  # segment distance from spherical geometry, adjusted for elevation
        dist_sph2d = [0]  # cumulative distance from spherical geometry only
        dist_sph3d = [0]  # cumulative distance from spherical geometry, adjusted for elevation
        delta_geo2d = [0]  # segment distance from geodesic method only
        delta_geo3d = [0]  # segment distance from geodesic method, adjusted for elevation
        dist_geo2d = [0]  # cumulative distance from geodesic method only
        dist_geo3d = [0]  # cumulative distance from geodesic method, adjusted for elevation

        for idx in range(1, len(points)):
            start = points[idx - 1]
            end = points[idx]

            # elevation
            temp_delta_elev = end.elevation - start.elevation
            delta_elev.append(temp_delta_elev)

            # time
            temp_delta_time = (end.time - start.time).total_seconds()
            delta_time.append(temp_delta_time)

            # distance from spherical model
            temp_delta_sph2d = distance.great_circle((start.latitude, start.longitude), (end.latitude, end.longitude)).m
            delta_sph2d.append(temp_delta_sph2d)
            dist_sph2d.append(dist_sph2d[-1] + temp_delta_sph2d)
            temp_delta_sph3d = sqrt(temp_delta_sph2d ** 2 + temp_delta_elev ** 2)
            delta_sph3d.append(temp_delta_sph3d)
            dist_sph3d.append(dist_sph3d[-1] + temp_delta_sph3d)

            # distance from geodesic model
            temp_delta_geo2d = distance.distance((start.latitude, start.longitude), (end.latitude, end.longitude)).m
            delta_geo2d.append(temp_delta_geo2d)
            dist_geo2d.append(dist_geo2d[-1] + temp_delta_geo2d)
            temp_delta_geo3d = sqrt(temp_delta_geo2d ** 2 + temp_delta_elev ** 2)
            delta_geo3d.append(temp_delta_geo3d)
            dist_geo3d.append(dist_geo3d[-1] + temp_delta_geo3d)

        # dump the lists into the dataframe
        self.df['delta_elev'] = delta_elev
        self.df['delta_time'] = delta_time
        self.df['delta_sph2d'] = delta_sph2d
        self.df['delta_sph3d'] = delta_sph3d
        self.df['dist_sph2d'] = dist_sph2d
        self.df['dist_sph3d'] = dist_sph3d
        self.df['delta_geo2d'] = delta_geo2d
        self.df['delta_geo3d'] = delta_geo3d
        self.df['dist_geo2d'] = dist_geo2d
        self.df['dist_geo3d'] = dist_geo3d

        # Stats
        print("Spherical Distance 2D: {:.3f} km".format(dist_sph2d[-1] / 1000))
        print("Spherical Distance 3D: {:.3f} km".format(dist_sph3d[-1] / 1000))
        print("Elevation Correction: {:.3f} meters".format((dist_sph3d[-1]) - (dist_sph2d[-1])))
        print("Geodesic Distance 2D: {:.3f} km".format(dist_geo2d[-1] / 1000))
        print("Geodesic Distance 3D: {:.3f} km".format(dist_geo3d[-1] / 1000))
        print("Elevation Correction: {:.3f} meters".format((dist_geo3d[-1]) - (dist_geo2d[-1])))
        print("Model Difference: {:.3f} meters".format((dist_geo3d[-1]) - (dist_sph3d[-1])))
        print("Total Time: {}".format(str(datetime.timedelta(seconds=sum(delta_time)))))
        print(f"Elevation Gain: {round(sum(self.df[self.df['delta_elev'] > 0]['delta_elev']), 2)}")
        print(f"Elevation Loss: {round(sum(self.df[self.df['delta_elev'] < 0]['delta_elev']), 2)}")

        if 0:
            fig_5 = px.line(self.df, x='time', y='dist_geo3d', template='plotly_dark')
            fig_5.show()

        self.df['inst_mps'] = self.df['delta_geo3d'] / self.df['delta_time']
        if 0:
            fig_8 = px.histogram(self.df, x='inst_mps', template='plotly_dark')
            fig_8.update_traces(xbins=dict(start=0, end=12, size=0.1))
            fig_8.show()

        self.df.fillna(0,
                       inplace=True)  # fill in the NaN's in the first row of distances and deltas with 0. They were breaking the overall average speed calculation

        self.df_moving = self.df[self.df[
                                     'inst_mps'] >= 0.9]  # make a new dataframe filtered to only records where instantaneous speed was greater than 0.9m/s

        avg_mps = (sum((self.df['inst_mps'] * self.df['delta_time'])) / sum(self.df['delta_time']))

        avg_mov_mps = (sum((self.df_moving['inst_mps'] * self.df_moving['delta_time'])) / sum(
            self.df_moving['delta_time']))

        print("Maximum Speed: {} km/h".format(round((3.6 * self.df['inst_mps'].max(axis=0)), 2)))
        print("Average Speed: {} km/h".format(round((3.6 * avg_mps), 2)))
        print("Average Moving Speed: {} km/h".format(round((3.6 * avg_mov_mps), 2)))
        print("Moving Time: {}".format(str(datetime.timedelta(seconds=sum(self.df_moving['delta_time'])))))

        self.df['avg_mps_roll'] = self.df['inst_mps'].rolling(20, center=True).mean()

        if 0:
            fig_16 = px.line(self.df, x='time', y=['inst_mps', 'avg_mps_roll'],
                             template='plotly_dark')  # as of 2020-05-26 Plotly 4.8 you can pass a list of columns to either x or y and plotly will figure it out

            fig_16.show()

        if 1:
            fig_5 = px.line(self.df, x='dist_geo2d', y='elev', template='plotly_dark')
            fig_5.show()

    def colorscale(self, g):
        halfn = 15

        a = np.arange((halfn - 0.5), -halfn, -1.0)
        red = np.append(halfn * [255], np.linspace(255, 0, (halfn + 1)))
        green = np.append(np.linspace(0, 255, (halfn + 1)), halfn * [255])
        blue = (2 * halfn + 1) * [0]

        i = np.digitize(g, a)
        return "rgb({},{},{})".format(int(red[i]), int(green[i]), int(blue[i]))

    def gradient(self, interval, resolution=500):

        df_climb = self.df[['lon', 'lat', 'dist_geo2d', 'elev']]
        df_climb = df_climb[(df_climb["dist_geo2d"] >= interval[0]) & (df_climb["dist_geo2d"] <= interval[1])]
        df_climb['dist_geo2d_neg'] = -(df_climb["dist_geo2d"].iloc[-1] - df_climb["dist_geo2d"])

        steps = np.arange(0, np.min(df_climb['dist_geo2d_neg']), -resolution)
        steps = np.append(steps, np.min(df_climb['dist_geo2d_neg']))

        for step in steps[1:-1]:
            df_climb = pd.concat([df_climb, pd.DataFrame(data={'dist_geo2d_neg': step}, index=[0])],
                                 axis=0, join='outer', ignore_index=True)

        df_climb = df_climb.sort_values(by='dist_geo2d_neg')
        df_climb = df_climb.interpolate(method='linear', limit_direction='backward', limit=1)

        df_gradient = df_climb[df_climb['dist_geo2d_neg'].isin(steps)]
        df_gradient['elev_delta'] = df_gradient.elev.diff().shift(-1)
        df_gradient['dist_delta'] = df_gradient.dist_geo2d_neg.diff().shift(-1)
        df_gradient['gradient'] = df_gradient['elev_delta'] / df_gradient['dist_delta'] * 100

        '''
        fig = go.Figure(
            data=[go.Bar(
                x=gradient_details_df['gradient_range'].astype(str),
                y=gradient_details_df['total_distance'].apply(lambda x: round(x / 1000, 2)),
                marker_color=colors
            )],
            layout=go.Layout(
                bargap=0,
                title='Gradient profile of a route',
                xaxis_title='Gradient range (%)',
                yaxis_title='Distance covered (km)',
                autosize=False,
                width=1440,
                height=800,
                template='simple_white'
            )
        )
        fig.show()
        '''

        fig_gradient = go.Figure()
        steps = np.flip(steps)
        for i in range(len(steps) - 1):
            portion = df_climb[(df_climb['dist_geo2d_neg'] >= steps[i]) & (df_climb['dist_geo2d_neg'] <= steps[i + 1])]
            g = df_gradient['gradient'].iloc[i]
            fig_gradient.add_trace(
                go.Scatter(x=portion['dist_geo2d_neg'], y=portion['elev'], fill='tozeroy', fillcolor=self.colorscale(g),
                           mode='none', name='', showlegend=False))
            fig_gradient.add_annotation(x=np.mean(portion['dist_geo2d_neg']), y=np.max(portion['elev']) + 10,
                                        text="{:.1f}%".format(g),
                                        showarrow=False,
                                        arrowhead=1)
        fig_gradient.show()

        if 0:
            fig_map = px.line_mapbox(df_climb, lat='lat', lon='lon', hover_name='dist_geo2d_neg', mapbox_style="open-street-map",
                                     zoom=11)
            fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
            fig_map.show()

        if 0:
            fig_map = px.scatter_mapbox(df_climb, lat="lat", lon="lon", hover_name="dist_geo2d_neg", zoom=11)
            fig_map.update_layout(mapbox_style="open-street-map")
            fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
            fig_map.show()

        if 1:
            print(np.mean(df_climb["lon"]))
            print(np.mean(df_climb["lat"]))

            fig_map = go.Figure(go.Scattermapbox(lat=df_climb["lat"],
                                                 lon=df_climb["lon"],
                                                 mode='markers',
                                                 marker=go.scattermapbox.Marker(size=6)))
            fig_map.update_layout(mapbox_style="open-street-map")
            fig_map.update_layout(
                hovermode='closest',
                mapbox=dict(
                    bearing=0,
                    center=go.layout.mapbox.Center(
                        lat=np.mean(df_climb["lat"]),
                        lon=np.mean(df_climb["lon"])
                    ),
                    pitch=0,
                    zoom=11
                )
            )
            fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
            fig_map.show()


alpe = Geppetto("tracks/The_missing_pass_W3_D2_.gpx")
alpe.gradient(interval=[33739, 48124])

# arcana = Geppetto("tracks/Local_passes_gravel_edition_.gpx")
# arcana.gradient(interval=[25510, 41000], resolution=1000)

# cirone = Geppetto("tracks/Cisa_e_Cirone.gpx")
# cirone.gradient(interval=[62380, 76542], resolution=500)
