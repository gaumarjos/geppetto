"""
DOCUMENTATION

Importing
https://towardsdatascience.com/parsing-fitness-tracker-data-with-python-a59e7dc17418

Maths
https://thatmaceguy.github.io/python/gps-data-analysis-intro/
https://rkurchin.github.io/posts/2020/05/ftp

Graphics
https://plotly.com/python/mapbox-layers/
https://plotly.com/python/builtin-colorscales/
https://github.com/plotly/plotly.py/issues/1728
https://plotly.com/python/filled-area-plots/
"""

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
import scipy.constants as const
import os
import fitdecode
from itertools import product


class Geppetto():
    def __init__(self, files, plots=False, debug_plots=False, debug=False):
        """
        The object can load multiple gpx files at once. Useful when we want to plot multiple traces on the same map. The
        processing is done independently using local variables and the results are then appended to a class variable.
        All speeds are in m/s unless km/h are specified (and that can happen only in plots)

        :param files: an array of .gpx files
        """

        # Replacement value
        default_atemp = 26  # np.nan

        def import_gpx(file):
            """
            Import a gpx file
            :param file: filename
            :return: Pandas Dataframe
            """
            gpx = gpxpy.parse(open(file, 'r'))
            points = gpx.tracks[0].segments[0].points
            local_df = pd.DataFrame(columns=['time', 'lon', 'lat', 'elev', 'atemp', 'hr', 'cad'])
            for point in points:
                # See what extension tags are there
                atemp = None
                hr = None
                cad = None
                for ext in point.extensions:
                    for extchild in list(ext):
                        if extchild.tag == "{http://www.garmin.com/xmlschemas/TrackPointExtension/v1}atemp":
                            atemp = float(extchild.text)
                        elif extchild.tag == "{http://www.garmin.com/xmlschemas/TrackPointExtension/v1}hr":
                            hr = float(extchild.text)
                        elif extchild.tag == "{http://www.garmin.com/xmlschemas/TrackPointExtension/v1}cad":
                            cad = float(extchild.text)
                # Add data to the df
                local_df = pd.concat([local_df, pd.DataFrame(data={'time': point.time,
                                                                   'lon': point.longitude,
                                                                   'lat': point.latitude,
                                                                   'elev': point.elevation,
                                                                   'atemp': atemp if atemp is not None else default_atemp,
                                                                   'hr': hr if hr is not None else np.nan,
                                                                   'cad': cad if cad is not None else np.nan,
                                                                   },
                                                             index=[0])],
                                     axis=0,
                                     join='outer',
                                     ignore_index=True)
            return local_df

        def import_fit(file):
            """
            Import a FIT file generated from a Garmin device
            :param file: filename
            :return: Pandas Dataframe

            Fields:
                timestamp
                position_lat
                position_long
                distance
                enhanced_altitude
                altitude
                enhanced_speed
                speed
                unknown_87
                cadence
                temperature
                fractional_cadence
            """

            with fitdecode.FitReader(file) as fit:

                local_df = pd.DataFrame(
                    columns=['time', 'lon', 'lat', 'dist', 'elev', 'enhanced_elev', 'speed', 'enhanced_speed', 'atemp',
                             'hr',
                             'cad', 'fractional_cad'])

                for frame in fit:
                    if isinstance(frame, fitdecode.records.FitDataMessage):
                        if frame.name == 'lap':
                            # This frame contains data about a lap.
                            '''
                            for field in frame.fields:
                                # field is a FieldData object
                                print(field.name)
                            '''
                            pass

                        elif frame.name == 'record':
                            # This frame contains data about a "track point".
                            '''
                            for field in frame.fields:
                                # field is a FieldData object
                                print(field.name)
                            '''
                            time = frame.get_value('timestamp', fallback=np.nan)
                            lon = float(frame.get_value('position_long', fallback=np.nan)) / ((2 ** 32) / 360)
                            lat = float(frame.get_value('position_lat', fallback=np.nan)) / ((2 ** 32) / 360)
                            dist = frame.get_value('distance', fallback=np.nan)
                            elev = frame.get_value('altitude', fallback=np.nan)
                            enhanced_elev = frame.get_value('enhanced_altitude', fallback=np.nan)
                            speed = frame.get_value('speed', fallback=np.nan)
                            enhanced_speed = frame.get_value('enhanced_speed', fallback=np.nan)
                            atemp = frame.get_value('temperature', fallback=default_atemp)
                            hr = frame.get_value('heart_rate', fallback=np.nan)
                            cad = frame.get_value('cadence', fallback=np.nan)
                            fractional_cad = frame.get_value('fractional_cadence', fallback=np.nan)

                            local_df = pd.concat([local_df, pd.DataFrame(data={'time': time,
                                                                               'lon': lon,
                                                                               'lat': lat,
                                                                               'dist': dist,
                                                                               'elev': elev,
                                                                               'enhanced_elev': enhanced_elev,
                                                                               'speed': speed,
                                                                               'enhanced_speed': enhanced_speed,
                                                                               'atemp': atemp,
                                                                               'hr': hr,
                                                                               'cad': cad,
                                                                               'fractional_cad': fractional_cad,
                                                                               },
                                                                         index=[0])],
                                                 axis=0,
                                                 join='outer',
                                                 ignore_index=True)

            return local_df

        # Create empty lists to be populated
        self.df = []  # "raw" traces
        self.df_moving = []  # traces with idle segments removed

        # Go through all files
        for file in files:
            print("-------- Filename: {} --------".format(file))

            _, extension = os.path.splitext(file)
            if extension == ".gpx":
                df = import_gpx(file)
            elif extension == ".fit":
                df = import_fit(file)
            else:
                print("Not a recognised file format, skipping.")
                continue

            if debug:
                print(df)
                print(df.describe())

            if debug_plots:
                fig_2 = px.scatter_3d(df, x='lon', y='lat', z='elev', color='elev', template='plotly_dark')
                fig_2.update_traces(marker=dict(size=2), selector=dict(mode='markers'))
                fig_2.show()

            # first create lists to store the results, these will be appended ot the dataframe at the end
            # Note: i'll be working from the gps_points object directly and then appending results into the dataframe. It would make a lot more sense to operate directly on the dataframe.

            c_delta_elev = [0]  # change in elevation between records
            c_delta_time = [0]  # time interval between records
            c_delta_sph2d = [0]  # segment distance from spherical geometry only
            c_delta_sph3d = [0]  # segment distance from spherical geometry, adjusted for elevation
            c_dist_sph2d = [0]  # cumulative distance from spherical geometry only
            c_dist_sph3d = [0]  # cumulative distance from spherical geometry, adjusted for elevation
            c_delta_geo2d = [0]  # segment distance from geodesic method only
            c_delta_geo3d = [0]  # segment distance from geodesic method, adjusted for elevation
            c_dist_geo2d = [0]  # cumulative distance from geodesic method only
            c_dist_geo3d = [0]  # cumulative distance from geodesic method, adjusted for elevation

            for idx in range(1, len(df)):
                start = df.iloc[idx - 1]
                end = df.iloc[idx]

                # elevation
                temp_c_delta_elev = end['elev'] - start['elev']
                c_delta_elev.append(temp_c_delta_elev)

                # time
                temp_c_delta_time = (end.time - start.time).total_seconds()
                c_delta_time.append(temp_c_delta_time)

                # distance from spherical model
                temp_c_delta_sph2d = distance.great_circle((start.lat, start.lon),
                                                           (end.lat, end.lon)).m
                c_delta_sph2d.append(temp_c_delta_sph2d)
                c_dist_sph2d.append(c_dist_sph2d[-1] + temp_c_delta_sph2d)
                temp_c_delta_sph3d = sqrt(temp_c_delta_sph2d ** 2 + temp_c_delta_elev ** 2)
                c_delta_sph3d.append(temp_c_delta_sph3d)
                c_dist_sph3d.append(c_dist_sph3d[-1] + temp_c_delta_sph3d)

                # distance from geodesic model
                temp_c_delta_geo2d = distance.distance((start.lat, start.lon), (end.lat, end.lon)).m
                c_delta_geo2d.append(temp_c_delta_geo2d)
                c_dist_geo2d.append(c_dist_geo2d[-1] + temp_c_delta_geo2d)
                temp_c_delta_geo3d = sqrt(temp_c_delta_geo2d ** 2 + temp_c_delta_elev ** 2)
                c_delta_geo3d.append(temp_c_delta_geo3d)
                c_dist_geo3d.append(c_dist_geo3d[-1] + temp_c_delta_geo3d)

            # dump the lists into the dataframe
            df['c_delta_elev'] = c_delta_elev
            df['c_delta_time'] = c_delta_time
            df['c_delta_sph2d'] = c_delta_sph2d
            df['c_delta_sph3d'] = c_delta_sph3d
            df['c_dist_sph2d'] = c_dist_sph2d
            df['c_dist_sph3d'] = c_dist_sph3d
            df['c_delta_geo2d'] = c_delta_geo2d
            df['c_delta_geo3d'] = c_delta_geo3d
            df['c_dist_geo2d'] = c_dist_geo2d
            df['c_dist_geo3d'] = c_dist_geo3d

            # Stats
            print("Spherical Distance 2D: {:.3f} km".format(c_dist_sph2d[-1] / 1000))
            print("Spherical Distance 3D: {:.3f} km".format(c_dist_sph3d[-1] / 1000))
            print("Elevation Correction (spherical 3D-2D): {:.0f} m".format((c_dist_sph3d[-1]) - (c_dist_sph2d[-1])))
            print("Geodesic Distance 2D: {:.3f} km".format(c_dist_geo2d[-1] / 1000))
            print("Geodesic Distance 3D: {:.3f} km".format(c_dist_geo3d[-1] / 1000))
            print("Elevation Correction (geodesic 3D-2D): {:.0f} m".format((c_dist_geo3d[-1]) - (c_dist_geo2d[-1])))
            print("Model Difference (spherical-geodesic 3D): {:.0f} m".format((c_dist_geo3d[-1]) - (c_dist_sph3d[-1])))
            print("Total Time: {}".format(str(datetime.timedelta(seconds=sum(c_delta_time)))))
            print(f"Elevation Gain: {round(sum(df[df['c_delta_elev'] > 0]['c_delta_elev']), 2)}")
            print(f"Elevation Loss: {round(sum(df[df['c_delta_elev'] < 0]['c_delta_elev']), 2)}")

            if debug_plots:
                fig_3 = px.line(df, x='time', y='c_dist_geo3d', template='plotly_dark')
                fig_3.show()

            # Speed
            df['c_speed'] = df['c_delta_geo3d'] / df['c_delta_time']

            # Remove absurd outliers
            df = df[df['c_speed'] < 80.0 / 3.6]

            # Check and fill nan's
            if df.isna().sum().sum() > 1:
                print("Warning: too many NaN's")
            df.fillna(0, inplace=True)

            # Look at c_delta_geo3d and c_delta_time over time and see if everything makes sense
            if 0:
                fig_c_delta_geo3d = px.line(df, x='time', y='c_delta_geo3d')
                fig_c_delta_geo3d.show()
                fig_c_delta_time = px.line(df, x='time', y='c_delta_time')
                fig_c_delta_time.show()
                fig_c_speed = px.line(df, x='time', y='c_speed')
                fig_c_speed.show()

            # Speed distribution to determine cutoff speed to remove idle points. Threshold determined to be 0.9 m/s.
            if debug_plots:
                fig_4 = px.histogram(df, x='c_speed', template='plotly_dark')
                fig_4.update_traces(xbins=dict(start=0, end=12, size=0.1))
                fig_4.show()

            # Remove idle points and compare average speeds
            df_moving = df[df['c_speed'] >= 0.9]
            avg_speed = sum((df['c_speed'] * df['c_delta_time'])) / sum(df['c_delta_time'])
            avg_mov_speed = sum((df_moving['c_speed'] * df_moving['c_delta_time'])) / sum(
                df_moving['c_delta_time'])
            print("Maximum Speed: {} km/h".format(round((3.6 * df['c_speed'].max(axis=0)), 2)))
            print("Average Speed: {} km/h".format(round((3.6 * avg_speed), 2)))
            print("Average Moving Speed: {} km/h".format(round((3.6 * avg_mov_speed), 2)))
            print("Moving Time: {}".format(str(datetime.timedelta(seconds=sum(df_moving['c_delta_time'])))))

            # Coarsely filter speed to remove outliers (this is a bad way to do it, a Kalman filter should be used)
            df['avg_speed_roll'] = df['c_speed'].rolling(20, center=True).mean()
            if debug_plots:
                fig_5 = px.line(df, x='time', y=['c_speed', 'avg_speed_roll'],
                                template='plotly_dark')
                fig_5.show()

            if debug_plots:
                fig_5 = px.line(df, x='time', y=['cad', 'atemp', 'hr'],
                                template='plotly_dark')
                fig_5.show()

            if debug_plots:
                fig_5 = px.scatter(df, x='c_speed', y='cad', color='hr')
                fig_5.show()

            if debug_plots:
                fig_4 = px.histogram(df, x='cad', template='plotly_dark')
                fig_4.update_traces(xbins=dict(start=10, end=120, size=1))
                fig_4.show()

            # Once done, take the current df (local) and append it to the list of df's
            self.df.append(df)
            self.df_moving.append(df_moving)
            print()

        # Map and elevation plots
        if plots:
            fig_map = go.Figure()
            for i, df in enumerate(self.df):
                fig_map.add_trace(go.Scattermapbox(lat=df["lat"],
                                                   lon=df["lon"],
                                                   mode='lines+markers',
                                                   marker=go.scattermapbox.Marker(size=6),
                                                   name=files[i],
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
                fig_elev.add_trace(go.Scatter(x=df["c_dist_geo2d"],
                                              y=df["elev"],
                                              mode='lines+markers',
                                              name=files[i],
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

        # Select only the points belonging to the climb
        assert interval[0] >= 0
        assert interval[1] >= 0
        df_climb = self.df[trace_nr][['lon', 'lat', 'c_dist_geo2d', 'elev']].copy()
        if interval[1] == 0:
            df_climb = df_climb[df_climb["c_dist_geo2d"] >= interval[0]]
        else:
            df_climb = df_climb[(df_climb["c_dist_geo2d"] >= interval[0]) & (df_climb["c_dist_geo2d"] <= interval[1])]
        df_climb['c_dist_geo2d_neg'] = -(df_climb["c_dist_geo2d"].iloc[-1] - df_climb["c_dist_geo2d"])

        steps = np.arange(0, np.min(df_climb['c_dist_geo2d_neg']), -resolution)
        steps = np.append(steps, np.min(df_climb['c_dist_geo2d_neg']))

        for step in steps[1:-1]:
            df_climb = pd.concat([df_climb, pd.DataFrame(data={'c_dist_geo2d_neg': step}, index=[0])],
                                 axis=0, join='outer', ignore_index=True)

        df_climb = df_climb.sort_values(by='c_dist_geo2d_neg')
        df_climb = df_climb.interpolate(method='linear', limit_direction='backward', limit=1)

        # Copy only elements at step distance in a new df
        df_climb_subset = df_climb[df_climb['c_dist_geo2d_neg'].isin(steps)].copy()
        df_climb_subset['c_elev_delta'] = df_climb_subset.elev.diff().shift(-1)
        df_climb_subset['c_dist_delta'] = df_climb_subset.c_dist_geo2d_neg.diff().shift(-1)
        df_climb_subset['c_gradient'] = df_climb_subset['c_elev_delta'] / df_climb_subset['c_dist_delta'] * 100

        # Option #1
        # Plots in independent pages
        if 0:
            # Gradient
            fig_gradient = go.Figure()
            steps = np.flip(steps)
            for i in range(len(steps) - 1):
                portion = df_climb[
                    (df_climb['c_dist_geo2d_neg'] >= steps[i]) & (df_climb['c_dist_geo2d_neg'] <= steps[i + 1])]
                g = df_climb_subset['c_gradient'].iloc[i]
                fig_c_gradient.add_trace(
                    go.Scatter(x=portion['c_dist_geo2d_neg'], y=portion['elev'], fill='tozeroy',
                               fillcolor=self.colorscale(g),
                               mode='none', name='', showlegend=False))
                fig_c_gradient.add_annotation(x=np.mean(portion['c_dist_geo2d_neg']), y=np.max(portion['elev']) + 10,
                                              text="{:.1f}%".format(g),
                                              showarrow=False,
                                              arrowhead=1)
            fig_c_gradient.show()

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
                    (df_climb['c_dist_geo2d_neg'] >= steps[i]) & (df_climb['c_dist_geo2d_neg'] <= steps[i + 1])]
                g = df_climb_subset['c_gradient'].iloc[i]
                fig.add_trace(go.Scatter(x=portion['c_dist_geo2d_neg'],
                                         y=portion['elev'],
                                         fill='tozeroy',
                                         fillcolor=self.colorscale(g),
                                         mode='none',
                                         name='',
                                         showlegend=False),
                              row=1,
                              col=1
                              )
                fig.add_annotation(x=np.mean(portion['c_dist_geo2d_neg']), y=np.max(portion['elev']) + 10,
                                   text="{:.1f}".format(g),
                                   showarrow=True,
                                   arrowhead=0)

            # Map (1,2)
            fig.add_trace(go.Scattermapbox(lat=df_climb["lat"],
                                           lon=df_climb["lon"],
                                           mode='lines+markers',
                                           line=dict(
                                               width=2,
                                               color="gray",
                                           ),
                                           marker=go.scattermapbox.Marker(size=6,
                                                                          color=df_climb["elev"],
                                                                          colorscale=px.colors.sequential.Bluered),
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

    def cadence_speed_curve(self, interval, trace_nr=0):
        """
        This method operates on only one trace
        :param interval: [start_meter, end_meter]
        :param trace_nr: in case multiple files are loaded at init, whose we want to compute the gradient
        :return: plots
        """

        # Select only the points belonging to the climb
        assert interval[0] >= 0
        assert interval[1] >= 0
        df_climb = self.df[trace_nr][['lon', 'lat', 'c_dist_geo2d', 'elev', 'cad', 'c_speed', 'hr']].copy()
        if interval[1] == 0:
            df_climb = df_climb[df_climb["c_dist_geo2d"] >= interval[0]]
        else:
            df_climb = df_climb[(df_climb["c_dist_geo2d"] >= interval[0]) & (df_climb["c_dist_geo2d"] <= interval[1])]

        cadence = np.linspace(0, 110, 100)
        gears = [50. / 11., 50. / 12., 50. / 13, 50. / 14., 50. / 16., 50. / 18, 50. / 20., 50. / 22., 50. / 25,
                 50. / 28., 50. / 32., 34. / 11., 34. / 12., 34. / 13, 34. / 14., 34. / 16., 34. / 18, 34. / 20.,
                 34. / 22., 34. / 25, 34. / 28., 34. / 32.]

        # Plot
        fig_cadence = go.Figure()
        fig_cadence.add_trace(go.Scatter(x=df_climb["cad"],
                                         y=df_climb["c_speed"],
                                         mode='markers',
                                         name="Measured",
                                         marker=go.scatter.Marker(size=6,
                                                                  color=df_climb["hr"],
                                                                  colorscale=px.colors.sequential.Bluered),
                                         )
                              )
        for gear in gears:
            speed = 2.0 * np.pi * (622.0 / 1000 / 2.0) * gear * (cadence / 60.0)  # mps
            fig_cadence.add_trace(go.Scatter(x=cadence,
                                             y=speed,
                                             mode='lines',
                                             name="{:.1f}".format(gear),
                                             line=dict(
                                                 width=1,
                                                 color="gray"),
                                             )
                                  )
        fig_cadence.update_xaxes(range=[40, 120])
        fig_cadence.update_yaxes(range=[0, 50. / 3.6])
        fig_cadence.update_layout(title="Cadence - Speed curve")
        fig_cadence.update_layout(margin={"r": 40, "t": 40, "l": 40, "b": 40})
        fig_cadence.show()

    def estimate_power(self, interval, trace_nr=0):
        """
        This method operates on only one trace
        :param interval: [start_meter, end_meter]
        :param trace_nr: in case multiple files are loaded at init, whose we want to compute the gradient
        :return: plots
        """

        def P_air(speed, altitude, T_C, CdA=0.28, L_dt=0.051):
            """
            :param speed: [m/s]
            :param altitude: [m]
            :param T: [K]
            :param CdA:
            :return: [N]
            """
            # Air density
            L = 0.0065
            T0 = 298
            M = 0.02896
            Rs = 287.058
            T_K = T_C + 273.15
            p_exp = M * const.g / (const.R * L)
            p = const.atm * (1 - (L * altitude / T0)) ** p_exp
            rho = p / (Rs * T_K)  # kg/m^3
            F_air = 0.5 * CdA * rho * speed ** 2
            P_air = F_air * speed / (1.0 - L_dt)
            return P_air

        def P_roll(gradient, m, speed, Crr=0.00321, L_dt=0.051):
            """
            Estimate rolling resistance for the 2 tires (TBC)
            :param grad: ratio
            :param m: [kg]
            :param Crr: 0.00321 for Continental GP5000 @6.9bar
            :return: [N]
            """
            F_roll = 2. * Crr * np.cos(np.arctan(gradient)) * m * const.g
            P_roll = F_roll * speed / (1.0 - L_dt)
            return P_roll

        def P_grav(gradient, m, speed, L_dt=0.051):
            """
            :param gradient: ratio
            :param m: [kg]
            :return: [N]
            """
            F_grav = np.sin(np.arctan(gradient)) * m * const.g
            P_grav = F_grav * speed / (1.0 - L_dt)
            return P_grav

        # Work on a portion of the track
        assert interval[0] >= 0
        assert interval[1] >= 0
        df_selection = self.df[trace_nr][['time', 'lon', 'lat', 'c_dist_geo2d', 'elev', 'c_speed', 'atemp']].copy()
        if interval[1] == 0:
            df_selection = df_selection[df_selection["c_dist_geo2d"] >= interval[0]]
        else:
            df_selection = df_selection[
                (df_selection["c_dist_geo2d"] >= interval[0]) & (df_selection["c_dist_geo2d"] <= interval[1])]

        # Compute gradient
        df_selection['c_elev_delta'] = df_selection.elev.diff().shift(-1)
        df_selection['c_dist_delta'] = df_selection.c_dist_geo2d.diff().shift(-1)
        df_selection['c_gradient'] = df_selection['c_elev_delta'] / df_selection['c_dist_delta']

        if 0:
            fig_5 = px.line(df_selection, x='c_dist_geo2d', y=['c_gradient', 'c_elev_delta', 'c_dist_delta'],
                            template='plotly_dark')
            fig_5.show()

        # Filter
        use_filter = 1
        if use_filter:
            window = 20
            df_selection['c_speed'] = df_selection['c_speed'].rolling(window, center=True).mean()
            df_selection['c_gradient'] = df_selection['c_gradient'].rolling(window, center=True).mean()

        # Compute power contributions
        df_selection["c_power_air"] = df_selection.apply(lambda x: P_air(x.c_speed,
                                                                         x.elev,
                                                                         x.atemp),
                                                         axis=1)
        df_selection["c_power_roll"] = df_selection.apply(lambda x: P_roll(x.c_gradient,
                                                                           86,
                                                                           x.c_speed),
                                                          axis=1)
        df_selection["c_power_grav"] = df_selection.apply(lambda x: P_grav(x.c_gradient,
                                                                           86,
                                                                           x.c_speed),
                                                          axis=1)
        df_selection["c_power"] = df_selection["c_power_air"] + df_selection["c_power_roll"] + df_selection[
            "c_power_grav"]

        df_selection.fillna(0, inplace=True)

        # Ideas
        # A) Consider positive power only
        # df_pushing = df_selection[df_selection['c_power'] >= 0.0]
        # B) Set all negative powers to zero
        df_selection.loc[df_selection["c_power"] < 0.0, "c_power"] = 0

        print("Average power: {} W".format(np.mean(df_selection['c_power'])))

        # Plot
        fig_power = go.Figure()
        fig_power.add_trace(go.Scatter(x=df_selection["c_dist_geo2d"],
                                       y=df_selection["c_power_grav"],
                                       name="Gravity",
                                       hoverinfo='x+y',
                                       mode='lines',
                                       line=dict(width=0.5, color='green'),
                                       stackgroup='one'
                                       ))
        fig_power.add_trace(go.Scatter(x=df_selection["c_dist_geo2d"],
                                       y=df_selection["c_power_air"],
                                       name="Air",
                                       hoverinfo='x+y',
                                       mode='lines',
                                       line=dict(width=0.5, color='lightblue'),
                                       stackgroup='one'  # define stack group
                                       ))
        fig_power.add_trace(go.Scatter(x=df_selection["c_dist_geo2d"],
                                       y=df_selection["c_power_roll"],
                                       name="Roll",
                                       hoverinfo='x+y',
                                       mode='lines',
                                       line=dict(width=0.5, color='black'),
                                       stackgroup='one'
                                       ))
        fig_power.show()


def main():
    """
    Main function
    :return: nothing
    """
    # geppetto = Geppetto(["tracks/The_missing_pass_W3_D2_.gpx",
    #                  "tracks/Local_passes_gravel_edition_.gpx",
    #                  "tracks/Two_more_W20_D3_.gpx",
    #                  "tracks/The_local_4_or_5_passes.gpx",
    #                  "tracks/More_local_passes_W17_D3_.gpx",
    #                  "tracks/More_local_4_passes.gpx",
    #                  "tracks/More_and_more_local_passes_W19_D3_.gpx",
    #                  "tracks/Even_more_local_passes.gpx",
    #                  "tracks/Cisa_e_Cirone.gpx",
    #                  "tracks/Autumnal_chestnut_trees_Cisa_and_Brattello.gpx",
    #                  ],
    #                 plots=True)

    alpe = Geppetto(["tracks/The_missing_pass_W3_D2_.gpx"], plots=False, debug=0, debug_plots=0)
    alpe.gradient(interval=[33739, 48124])
    # alpe.cadence_speed_curve(interval=[0, 0])
    alpe.estimate_power(interval=[33739, 48124])
    alpe.estimate_power(interval=[0, 0])

    # lagastrello = Geppetto(["tracks/More_local_4_passes.gpx"], plots=False, debug=0, debug_plots=0)
    # lagastrello.gradient(interval=[94819, 106882])
    # lagastrello.cadence_speed_curve(interval=[0, 0])
    # lagastrello.estimate_power(interval=[0, 0])

    # arcana = Geppetto(["tracks/Local_passes_gravel_edition_W2_D2_.fit"], plots=False, debug=1, debug_plots=0)
    # arcana.gradient(interval=[94819, 106882])
    # arcana.cadence_speed_curve(interval=[0, 0])
    # arcana.estimate_power(interval=[0, 0])


if __name__ == "__main__":
    main()
