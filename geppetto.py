'''
DOCUMENTATION

Maths
https://thatmaceguy.github.io/python/gps-data-analysis-intro/

Graphics
https://plotly.com/python/mapbox-layers/
https://plotly.com/python/builtin-colorscales/
https://github.com/plotly/plotly.py/issues/1728
'''

import numpy as np
import gpxpy
import gpxpy.gpx
import plotly.express as px
import pandas as pd
from geopy import distance
from math import sqrt, floor
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Geppetto():
    def __init__(self, gpx_files, plots=False, debug_plots=False, debug=False):
        """
        The object can load multiple gpx files at once. Useful when we want to plot multiple traces on the same map. The
        processing is done independently using local variables and the results are then appended to a class variable.
        :param gpx_files: an array of .gpx files
        """

        self.df = []

        for gpx_file in gpx_files:
            print("-------- Filename: {} --------".format(gpx_file))

            gpx = gpxpy.parse(open(gpx_file, 'r'))
            points = gpx.tracks[0].segments[0].points
            df = pd.DataFrame(columns=['lon', 'lat', 'elev', 'time'])
            for point in points:
                df = pd.concat([df, pd.DataFrame(data={'lon': point.longitude,
                                                       'lat': point.latitude,
                                                       'elev': point.elevation,
                                                       'time': point.time},
                                                 index=[0])],
                               axis=0,
                               join='outer',
                               ignore_index=True)

            if debug:
                print(df)
                print(df.describe())

            if debug_plots:
                fig_1 = px.scatter(df, x='lon', y='lat', template='plotly_dark')
                fig_1.show()

            if debug_plots:
                fig_2 = px.scatter_3d(df, x='lon', y='lat', z='elev', color='elev', template='plotly_dark')
                fig_2.update_traces(marker=dict(size=2), selector=dict(mode='markers'))
                fig_2.show()

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
                temp_delta_sph2d = distance.great_circle((start.latitude, start.longitude),
                                                         (end.latitude, end.longitude)).m
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
            df['delta_elev'] = delta_elev
            df['delta_time'] = delta_time
            df['delta_sph2d'] = delta_sph2d
            df['delta_sph3d'] = delta_sph3d
            df['dist_sph2d'] = dist_sph2d
            df['dist_sph3d'] = dist_sph3d
            df['delta_geo2d'] = delta_geo2d
            df['delta_geo3d'] = delta_geo3d
            df['dist_geo2d'] = dist_geo2d
            df['dist_geo3d'] = dist_geo3d

            # Stats
            print("Spherical Distance 2D: {:.3f} km".format(dist_sph2d[-1] / 1000))
            print("Spherical Distance 3D: {:.3f} km".format(dist_sph3d[-1] / 1000))
            print("Elevation Correction: {:.3f} meters".format((dist_sph3d[-1]) - (dist_sph2d[-1])))
            print("Geodesic Distance 2D: {:.3f} km".format(dist_geo2d[-1] / 1000))
            print("Geodesic Distance 3D: {:.3f} km".format(dist_geo3d[-1] / 1000))
            print("Elevation Correction: {:.3f} meters".format((dist_geo3d[-1]) - (dist_geo2d[-1])))
            print("Model Difference: {:.3f} meters".format((dist_geo3d[-1]) - (dist_sph3d[-1])))
            print("Total Time: {}".format(str(datetime.timedelta(seconds=sum(delta_time)))))
            print(f"Elevation Gain: {round(sum(df[df['delta_elev'] > 0]['delta_elev']), 2)}")
            print(f"Elevation Loss: {round(sum(df[df['delta_elev'] < 0]['delta_elev']), 2)}")

            if debug_plots:
                fig_3 = px.line(df, x='time', y='dist_geo3d', template='plotly_dark')
                fig_3.show()

            df['inst_mps'] = df['delta_geo3d'] / df['delta_time']
            if debug_plots:
                fig_4 = px.histogram(df, x='inst_mps', template='plotly_dark')
                fig_4.update_traces(xbins=dict(start=0, end=12, size=0.1))
                fig_4.show()

            df.fillna(0,
                      inplace=True)  # fill in the NaN's in the first row of distances and deltas with 0. They were breaking the overall average speed calculation

            df_moving = df[df[
                               'inst_mps'] >= 0.9]  # make a new dataframe filtered to only records where instantaneous speed was greater than 0.9m/s

            avg_mps = (sum((df['inst_mps'] * df['delta_time'])) / sum(df['delta_time']))

            avg_mov_mps = (sum((df_moving['inst_mps'] * df_moving['delta_time'])) / sum(
                df_moving['delta_time']))

            print("Maximum Speed: {} km/h".format(round((3.6 * df['inst_mps'].max(axis=0)), 2)))
            print("Average Speed: {} km/h".format(round((3.6 * avg_mps), 2)))
            print("Average Moving Speed: {} km/h".format(round((3.6 * avg_mov_mps), 2)))
            print("Moving Time: {}".format(str(datetime.timedelta(seconds=sum(df_moving['delta_time'])))))

            df['avg_mps_roll'] = df['inst_mps'].rolling(20, center=True).mean()

            if debug_plots:
                fig_5 = px.line(df, x='time', y=['inst_mps', 'avg_mps_roll'],
                                template='plotly_dark')  # as of 2020-05-26 Plotly 4.8 you can pass a list of columns to either x or y and plotly will figure it out

                fig_5.show()

            # Once done, take the current df (local) and append it to the list of df's
            print()
            self.df.append(df)

        # Map and elevation plots
        if plots:
            fig_map = go.Figure()
            for i, df in enumerate(self.df):
                fig_map.add_trace(go.Scattermapbox(lat=df["lat"],
                                                   lon=df["lon"],
                                                   mode='markers',
                                                   marker=go.scattermapbox.Marker(size=6),
                                                   name=gpx_files[i],
                                                   )
                                  )
                fig_map.update_layout(
                    hovermode='closest',
                    mapbox=dict(
                        style="open-street-map",
                        bearing=0,
                        center=go.layout.mapbox.Center(
                            lat=np.mean(df["lat"]),
                            lon=np.mean(df["lon"])
                        ),
                        pitch=0,
                        zoom=11
                    )
                )
            fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
            fig_map.show()

            fig_elev = go.Figure()
            for i, df in enumerate(self.df):
                fig_elev.add_trace(go.Scatter(x=df["dist_geo2d"],
                                              y=df["elev"],
                                              name=gpx_files[i],
                                              )
                                   )
            fig_elev.update_layout(margin={"r": 0, "t": 20, "l": 0, "b": 0})
            fig_elev.show()

    @staticmethod
    def colorscale(g):
        halfn = 15

        a = np.arange((halfn - 0.5), -halfn, -1.0)
        red = np.append(halfn * [255], np.linspace(255, 0, (halfn + 1)))
        green = np.append(np.linspace(0, 255, (halfn + 1)), halfn * [255])
        blue = (2 * halfn + 1) * [0]

        i = np.digitize(g, a)
        return "rgb({},{},{})".format(int(red[i]), int(green[i]), int(blue[i]))

    def gradient(self, interval, resolution=500, trace_nr=0):
        """
        This method operates on only one trace
        :param interval: [start_meter, end_meter]
        :param resolution: the "step" in which the gradiane tis calculated/averaged, in meters
        :param trace_nr: in case multiple files are loaded at init, whose we want to compute the gradient
        :return: plots
        """

        df_climb = self.df[trace_nr][['lon', 'lat', 'dist_geo2d', 'elev']].copy()
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

        # Option #1
        # Plots in independent pages
        if 0:
            # Gradient
            fig_gradient = go.Figure()
            steps = np.flip(steps)
            for i in range(len(steps) - 1):
                portion = df_climb[
                    (df_climb['dist_geo2d_neg'] >= steps[i]) & (df_climb['dist_geo2d_neg'] <= steps[i + 1])]
                g = df_gradient['gradient'].iloc[i]
                fig_gradient.add_trace(
                    go.Scatter(x=portion['dist_geo2d_neg'], y=portion['elev'], fill='tozeroy',
                               fillcolor=self.colorscale(g),
                               mode='none', name='', showlegend=False))
                fig_gradient.add_annotation(x=np.mean(portion['dist_geo2d_neg']), y=np.max(portion['elev']) + 10,
                                            text="{:.1f}%".format(g),
                                            showarrow=False,
                                            arrowhead=1)
            fig_gradient.show()

            # Map
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

        # Option #2
        # Plots in the same page
        if 1:
            fig = make_subplots(
                rows=1, cols=2,
                column_widths=[0.6, 0.4]
            )

            # Gradient (1,1)
            steps = np.flip(steps)
            for i in range(len(steps) - 1):
                portion = df_climb[
                    (df_climb['dist_geo2d_neg'] >= steps[i]) & (df_climb['dist_geo2d_neg'] <= steps[i + 1])]
                g = df_gradient['gradient'].iloc[i]
                fig.add_trace(go.Scatter(x=portion['dist_geo2d_neg'],
                                         y=portion['elev'],
                                         fill='tozeroy',
                                         fillcolor=self.colorscale(g),
                                         mode='none',
                                         name='',
                                         showlegend=False),
                              row=1,
                              col=1
                              )
                fig.add_annotation(x=np.mean(portion['dist_geo2d_neg']), y=np.max(portion['elev']) + 10,
                                   text="{:.1f}".format(g),
                                   showarrow=True,
                                   arrowhead=0)

            # Map (1,2)
            fig.add_trace(go.Scattermapbox(lat=df_climb["lat"],
                                           lon=df_climb["lon"],
                                           mode='markers',
                                           marker=go.scattermapbox.Marker(size=6,
                                                                          color=df_climb["elev"],
                                                                          colorscale=px.colors.sequential.Greens_r),
                                           subplot='mapbox2',
                                           name='',
                                           showlegend=False
                                           )
                          )
            fig.update_layout(
                hovermode='closest',
                mapbox2=dict(
                    style="open-street-map",
                    domain={'x': [0.55, 1.0], 'y': [0, 1]},
                    bearing=0,
                    center=go.layout.mapbox.Center(
                        lat=np.mean(df_climb["lat"]),
                        lon=np.mean(df_climb["lon"])
                    ),
                    pitch=0,
                    zoom=11
                )
            )
            fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

            fig.show()


def main():
    """
    Main function
    :return: nothing
    """
    alpe = Geppetto(["tracks/The_missing_pass_W3_D2_.gpx",
                     "tracks/Local_passes_gravel_edition_.gpx",
                     "tracks/Two_more_W20_D3_.gpx",
                     "tracks/The_local_4_or_5_passes.gpx",
                     "tracks/More_local_passes_W17_D3_.gpx",
                     "tracks/More_local_4_passes.gpx",
                     "tracks/More_and_more_local_passes_W19_D3_.gpx",
                     "tracks/Even_more_local_passes.gpx",
                     "tracks/Cisa_e_Cirone.gpx",
                     "tracks/Autumnal_chestnut_trees_Cisa_and_Brattello.gpx",
                     ],
                    plots=True)
    # alpe.gradient(interval=[33739, 48124])


if __name__ == "__main__":
    main()
