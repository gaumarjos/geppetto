"""
DOCUMENTATION

Importing
https://towardsdatascience.com/parsing-fitness-tracker-data-with-python-a59e7dc17418

Maths
https://thatmaceguy.github.io/python/gps-data-analysis-intro/
https://rkurchin.github.io/posts/2020/05/ftp

Plotly
https://plotly.com/python/mapbox-layers/
https://plotly.com/python/builtin-colorscales/
https://github.com/plotly/plotly.py/issues/1728
https://plotly.com/python/filled-area-plots/
https://plotly.com/python/mapbox-layers/#using-layoutmapboxlayers-to-specify-a-base-map

Dash
https://dash.plotly.com/interactive-graphing
"""

import numpy as np
import gpxpy
import gpxpy.gpx
import plotly.express as px
import pandas as pd
from geopy import distance
from math import sqrt
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.constants as const
import os
import fitdecode


def colorscale(gradient_value, half_n=15):
    """
    Generate the color shade relative to a specific gradient value in a green - yellow - red scale of 2*half_n+1
    values. If gradient value is beyond the extremes, the maximum values will be used.
    E.g.:
        gradient_value == -half_n --> green
        gradient_value == 0 --> yellow
        gradient_value == +half_n --> red
    :param gradient_value: the gradient value
    :param half_n: half of the number of shades minus 1 (the center value).
    :return: a RGB tuple
    """
    a = np.arange((half_n - 0.5), -half_n, -1.0)
    red = np.append(half_n * [255], np.linspace(255, 0, (half_n + 1)))
    green = np.append(np.linspace(0, 255, (half_n + 1)), half_n * [255])
    blue = (2 * half_n + 1) * [0]
    i = np.digitize(gradient_value, a)
    return "rgb({},{},{})".format(int(red[i]), int(green[i]), int(blue[i]))


class Geppetto:
    def __init__(self, files, debug_plots=False, debug=False, csv=False):
        """
        The object can load multiple gpx files at once. Useful when we want to plot multiple traces on the same map. The
        processing is done independently using local variables and the results are then appended to a class variable.
        All speeds are in m/s unless km/h are specified (and that can happen only in plots).
        If multiple files are loaded, an array of Dataframes is created.

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
        self.file = []

        # Go through all files
        for file in files:
            print()
            print("-------- Filename: {} --------".format(file))

            name, extension = os.path.splitext(file)
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
            if debug_plots:
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
            df['c_speed_filtered'] = df['c_speed'].rolling(20, center=True).mean()
            if debug_plots:
                fig_5 = px.line(df, x='time', y=['c_speed', 'c_speed_filtered'],
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
            self.file.append(file)

            # Save to CSV for further processing (it's going to be faster even than just reimporting a GPX)
            if csv:
                df.to_csv("{}.csv".format(name))

    def plot_map_elevation(self):
        """
        Plots all traces that were imported.
        :return: Plots
        """
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.75, 0.25]
        )
        for i, df in enumerate(self.df):
            fig.add_trace(go.Scattermapbox(lat=df["lat"],
                                           lon=df["lon"],
                                           mode='lines+markers',
                                           marker=go.scattermapbox.Marker(size=6),
                                           name=self.file[i],
                                           hovertext=df['c_dist_geo2d'],
                                           subplot='mapbox',
                                           )
                          )
            fig.update_layout(
                hovermode='closest',
                mapbox=dict(
                    style="open-street-map",
                    domain={'x': [0.0, 1.0], 'y': [0.25, 1.0]},
                    bearing=0,
                    center=go.layout.mapbox.Center(
                        lat=np.mean(df["lat"]),
                        lon=np.mean(df["lon"])
                    ),
                    pitch=0,
                    zoom=11
                )
            )

        for i, df in enumerate(self.df):
            fig.add_trace(go.Scatter(x=df["c_dist_geo2d"],
                                     y=df["elev"],
                                     mode='lines+markers',
                                     name=self.file[i],
                                     ),
                          row=2,
                          col=1
                          )
        fig.update_layout(margin={"r": 0, "t": 20, "l": 0, "b": 0})
        fig.show()

    def copy_segment(self, columns, interval=(0, 0), trace_nr=0):
        """
        Return a portion of one trace of the list of Dataframes
        :param columns: columns that will be copied, it makes no sense to copy all because that will complicate
        operations such as interpolate unnecessarily
        :param interval: extremes (included) of the segment, in m
        :param trace_nr: the position of the trace in the array of Dataframes
        :return:
        """
        # Select only the points belonging to the climb
        assert interval[0] >= 0
        assert interval[1] >= 0
        df_segment = self.df[trace_nr][columns].copy()
        if interval[1] == 0:
            df_segment = df_segment[df_segment["c_dist_geo2d"] >= interval[0]]
        else:
            df_segment = df_segment[
                (df_segment["c_dist_geo2d"] >= interval[0]) & (df_segment["c_dist_geo2d"] <= interval[1])]
        return df_segment

    def gradient(self, interval=(0, 0), resolution=1000, trace_nr=0):
        """
        Computes the gradient over a portion of one dataframe
        :param interval: the gradient is calculated over the portion [start_meter, end_meter] of the input trace
        :param resolution: the "step" in which the gradient tis calculated/averaged, in meters
        :param trace_nr: in case multiple files are loaded at init, whose we want to compute the gradient
        :return: two dataframes, the first with all elevation values and the second with gradient values at specific distances "resolution" meters apart.
        """

        # Create a local copy of the input array of dataframes containing only the points belonging to the portion,
        # usually a climb, of interest
        df_climb = self.copy_segment(columns=["lon", "lat", "c_dist_geo2d", "elev"],
                                     interval=interval,
                                     trace_nr=trace_nr)

        # Count distance backward from the end (top of the climb)
        df_climb['c_dist_geo2d_neg'] = -(df_climb["c_dist_geo2d"].iloc[-1] - df_climb["c_dist_geo2d"])

        # Add points corresponding to "round" steps
        steps = np.arange(0, np.min(df_climb['c_dist_geo2d_neg']), -resolution)
        steps = np.append(steps, np.min(df_climb['c_dist_geo2d_neg']))
        for step in steps[1:-1]:
            df_climb = pd.concat([df_climb, pd.DataFrame(data={'c_dist_geo2d_neg': step}, index=[0])],
                                 axis=0, join='outer', ignore_index=True)

        # Interpolate their values
        df_climb = df_climb.sort_values(by='c_dist_geo2d_neg')
        df_climb = df_climb.interpolate(method='linear', limit_direction='backward', limit=1)

        # Copy only elements at step distance in a new df
        df_climb_gradient = df_climb[df_climb['c_dist_geo2d_neg'].isin(steps)].copy()
        df_climb_gradient['c_elev_delta'] = df_climb_gradient.elev.diff().shift(-1)
        df_climb_gradient['c_dist_delta'] = df_climb_gradient.c_dist_geo2d_neg.diff().shift(-1)
        df_climb_gradient['c_gradient'] = df_climb_gradient['c_elev_delta'] / df_climb_gradient['c_dist_delta'] * 100

        # This columns is redundant but is useful to cross check that the filter worked well
        df_climb_gradient['steps'] = np.flip(steps)

        return df_climb, df_climb_gradient

    @staticmethod
    def plot_gradient(df, df_gradient):
        """

        :param df:
        :param df_gradient:
        :return:
        """
        fig = go.Figure()
        for i in range(len(df_gradient) - 1):
            portion = df[
                (df['c_dist_geo2d_neg'] >= df_gradient.iloc[i]["c_dist_geo2d_neg"]) & (
                        df['c_dist_geo2d_neg'] <= df_gradient.iloc[i + 1]["c_dist_geo2d_neg"])]
            g = df_gradient['c_gradient'].iloc[i]
            fig.add_trace(go.Scatter(x=portion['c_dist_geo2d_neg'],
                                     y=portion['elev'],
                                     fill='tozeroy',
                                     fillcolor=colorscale(g),
                                     mode='none',
                                     name='',
                                     showlegend=False),
                          # row=1,
                          # col=1
                          )
            fig.add_annotation(x=np.mean(portion['c_dist_geo2d_neg']), y=np.max(portion['elev']) + 10,
                               text="{:.1f}".format(g),
                               showarrow=False,
                               arrowhead=0)

        # Map (1,2)
        fig.add_trace(go.Scattermapbox(lat=df["lat"],
                                       lon=df["lon"],
                                       mode='lines+markers',
                                       hovertext=df["c_dist_geo2d_neg"],
                                       line=dict(
                                           width=2,
                                           color="gray",
                                       ),
                                       marker=go.scattermapbox.Marker(size=6,
                                                                      color=df["elev"],
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
                domain={'x': [0.66, 0.99], 'y': [0.01, 0.33]},
                bearing=0,
                center=go.layout.mapbox.Center(
                    lat=np.mean(df["lat"]),
                    lon=np.mean(df["lon"])
                ),
                pitch=0,
                zoom=11
            )
        )
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        fig.show()

    def estimate_power(self, interval=(0, 0), trace_nr=0, total_mass=76 + 1 + 1.5 + 8, debug_plots=False):
        """
        Estimates the power over a portion of one dataframe
        :param interval: [start_meter, end_meter]
        :param trace_nr: in case multiple files are loaded at init, whose we want to compute the gradient
        :param total_mass: bike + rider
        :param debug_plots: toggle intermediate debug plots
        :return: the same dataframe it was given as input (or a portion of it) with columns c_power_air,  c_power_roll,
                 c_power_grav and a total  c_power added.
        """

        def power_air(speed, altitude, temperature_degc, coefficient_drag_area=0.28, losses_drivetrain=0.051):
            """
            :param speed: [m/s]
            :param altitude: [m]
            :param temperature_degc: [°C]
            :param coefficient_drag_area: Coefficient of drag * area, to be estimated based
            :param losses_drivetrain: pedaling and drivetrain efficiency
            :return: [W]
            """
            # Air density
            L = 0.0065
            T0 = 298
            M = 0.02896
            Rs = 287.058
            temperature_kelvin = temperature_degc + 273.15
            p_exp = M * const.g / (const.R * L)
            p = const.atm * (1 - (L * altitude / T0)) ** p_exp
            rho = p / (Rs * temperature_kelvin)  # kg/m^3
            force_air = 0.5 * coefficient_drag_area * rho * speed ** 2
            return force_air * speed / (1.0 - losses_drivetrain)

        def power_roll(gradient, m, speed, coefficient_rollingresistance=0.00321, losses_drivetrain=0.051):
            """
            Estimate rolling resistance for the 2 tires (TBC)
            :param gradient: ratio
            :param m: [kg]
            :param speed: [m/s]
            :param coefficient_rollingresistance: 0.00321 for one Continental GP5000 @6.9bar
            :param losses_drivetrain: pedaling and drivetrain efficiency
            :return: [W]
            """
            force_roll = 2. * coefficient_rollingresistance * np.cos(np.arctan(gradient)) * m * const.g
            return force_roll * speed / (1.0 - losses_drivetrain)

        def power_grav(gradient, m, speed, losses_drivetrain=0.051):
            """
            :param gradient: ratio
            :param m: [kg]
            :param speed: [m/s]
            :param losses_drivetrain: pedaling and drivetrain efficiency
            :return: [W]
            """
            force_grav = np.sin(np.arctan(gradient)) * m * const.g
            return force_grav * speed / (1.0 - losses_drivetrain)

        # Work on a portion of the track
        df_selection = self.copy_segment(
            columns=['time', 'lon', 'lat', 'c_dist_geo2d', 'elev', 'c_speed', 'atemp', 'hr'],
            interval=interval,
            trace_nr=trace_nr)

        # Compute gradient
        df_selection['c_elev_delta'] = df_selection.elev.diff().shift(-1)
        df_selection['c_dist_delta'] = df_selection.c_dist_geo2d.diff().shift(-1)
        df_selection['c_gradient'] = df_selection['c_elev_delta'] / df_selection['c_dist_delta']

        if debug_plots:
            fig_5 = px.line(df_selection, x='c_dist_geo2d', y=['c_gradient', 'c_elev_delta', 'c_dist_delta'],
                            template='plotly_dark')
            fig_5.show()

        # Filter
        use_filter = 1
        if use_filter:
            window = 4
            df_selection['c_speed'] = df_selection['c_speed'].rolling(window, center=True).mean()
            df_selection['c_gradient'] = df_selection['c_gradient'].rolling(window, center=True).mean()

        # Compute power contributions
        df_selection["c_power_air"] = df_selection.apply(lambda x: power_air(x.c_speed,
                                                                             x.elev,
                                                                             x.atemp),
                                                         axis=1)
        df_selection["c_power_roll"] = df_selection.apply(lambda x: power_roll(x.c_gradient,
                                                                               total_mass,
                                                                               x.c_speed),
                                                          axis=1)
        df_selection["c_power_grav"] = df_selection.apply(lambda x: power_grav(x.c_gradient,
                                                                               total_mass,
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

        return df_selection

    @staticmethod
    def plot_power(df):
        # Plot
        # fig_power = go.Figure()
        fig_power = make_subplots(rows=2, cols=1, shared_xaxes=True)

        fig_power.add_trace(go.Scatter(x=df["c_dist_geo2d"],
                                       y=df["c_power_grav"],
                                       name="Gravity",
                                       hoverinfo='x+y',
                                       mode='lines',
                                       line=dict(width=0.0, color='green'),
                                       stackgroup='one'
                                       ),
                            row=1,
                            col=1)
        fig_power.add_trace(go.Scatter(x=df["c_dist_geo2d"],
                                       y=df["c_power_air"],
                                       name="Air",
                                       hoverinfo='x+y',
                                       mode='lines',
                                       line=dict(width=0.0, color='lightblue'),
                                       stackgroup='one'  # define stack group
                                       ),
                            row=1,
                            col=1)
        fig_power.add_trace(go.Scatter(x=df["c_dist_geo2d"],
                                       y=df["c_power_roll"],
                                       name="Roll",
                                       hoverinfo='x+y',
                                       mode='lines',
                                       line=dict(width=2.0, color='black'),
                                       stackgroup='one'
                                       ),
                            row=1,
                            col=1)
        fig_power.add_trace(go.Scatter(x=df["c_dist_geo2d"],
                                       y=df["hr"],
                                       name="HR",
                                       hoverinfo='x+y',
                                       mode='lines',
                                       line=dict(width=1.0, color='red'),
                                       ),
                            row=2,
                            col=1)
        fig_power.update_yaxes(showspikes=True)
        fig_power.update_xaxes(showspikes=True, title="Distance (m)")
        # fig_power.update_traces(xaxis="x")
        fig_power.show()

    def cadence_speed_curve(self, interval=(0, 0), trace_nr=0):
        """
        This method operates on only one trace
        :param interval: [start_meter, end_meter]
        :param trace_nr: in case multiple files are loaded at init, whose we want to compute the gradient
        :return: plots
        """

        # Work on a portion of the track
        df_selection = self.copy_segment(columns=['lon', 'lat', 'c_dist_geo2d', 'elev', 'cad', 'c_speed', 'hr'],
                                         interval=interval,
                                         trace_nr=trace_nr)

        cadence = np.linspace(0, 110, 100)
        gears = [50. / 11., 50. / 12., 50. / 13, 50. / 14., 50. / 16., 50. / 18, 50. / 20., 50. / 22., 50. / 25,
                 50. / 28., 50. / 32., 34. / 11., 34. / 12., 34. / 13, 34. / 14., 34. / 16., 34. / 18, 34. / 20.,
                 34. / 22., 34. / 25, 34. / 28., 34. / 32.]

        # Plot
        fig_cadence = go.Figure()
        fig_cadence.add_trace(go.Scatter(x=df_selection["cad"],
                                         y=df_selection["c_speed"],
                                         mode='markers',
                                         name="Measured",
                                         marker=go.scatter.Marker(size=6,
                                                                  color=df_selection["hr"],
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


def main():
    """
    Main function
    :return: nothing
    """

    if 0:
        geppetto = Geppetto(["tracks/The_missing_pass_W3_D2_.gpx",
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

    if 1:
        alpe = Geppetto(["tracks/The_missing_pass_W3_D2_.gpx"], debug=0, debug_plots=0, csv=1)
        # alpe.plot_map_elevation()
        # climb, climb_gradient = alpe.gradient(interval=[33739, 48124], resolution=500)
        # alpe.plot_gradient(climb, climb_gradient)

        df_power = alpe.estimate_power(interval=[33739, 48124])
        alpe.plot_power(df_power)
        # alpe.estimate_power(interval=[0, 0])

    if 0:
        nederland = Geppetto(["tracks/Nederland.gpx"], plots=0, debug=0, debug_plots=0)
        nederland.gradient()
        nederland.cadence_speed_curve()

    if 0:
        cagazzone = Geppetto(["tracks/CaCa.gpx"], plots=0, debug=0, debug_plots=0)
        cagazzone.gradient(interval=[7274, 12667], resolution=200)

    if 0:
        votigno = Geppetto(["tracks/Broletto_salita_di_Votigno_e_Canossa.gpx"], plots=0, debug=0, debug_plots=0)
        votigno.gradient(interval=[17676, 20572], resolution=200)

    if 0:
        lagastrello = Geppetto(["tracks/More_local_4_passes.gpx"], plots=False, debug=0, debug_plots=0)
        lagastrello.gradient(interval=[94819, 106882])
        lagastrello.cadence_speed_curve(interval=[0, 0])
        lagastrello.estimate_power(interval=[0, 0])

    if 0:
        arcana = Geppetto(["tracks/Local_passes_gravel_edition_W2_D2_.fit"], plots=False, debug=1, debug_plots=0)
        arcana.gradient(interval=[94819, 106882])
        arcana.cadence_speed_curve(interval=[0, 0])
        arcana.estimate_power(interval=[0, 0])


if __name__ == "__main__":
    main()
