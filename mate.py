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
from geopy.geocoders import Nominatim
import json
import gzip
import shutil
import pyarrow.feather as feather
from tqdm import tqdm

TRACK_LIST = "tracklist.json"


def scan_files(folder, verbose=False):
    if os.path.exists(TRACK_LIST):
        with open(TRACK_LIST, "r") as json_file:
            # Extract list from json
            data_in_json = json.load(json_file)
            l = []
            for item in data_in_json:
                l.append(item['value'])
            files_in_json = sorted(l)

            # List files in folder
            files_in_folder = sorted(os.listdir(folder))

            if verbose:
                print(files_in_json)
                print(files_in_folder)

            # Compare
            if files_in_json == files_in_folder:
                if verbose:
                    print("Use json")
                return data_in_json
            else:
                if verbose:
                    print("Update json")
                return files_location_info(folder=folder)
    else:
        if verbose:
            print("Create json")
        return files_location_info(folder=folder)


def files_location_info(folder):
    """
    Scan a folder
    :param folder:
    :return:
    """
    l = []
    for file in sorted(os.listdir(folder)):
        l.append({'label': "{} {}".format(file, file_location_info(folder + file)),
                  'value': file,
                  })

    # Dump result in json
    with open(TRACK_LIST, "w") as json_file:
        json.dump(l, json_file, indent=4)

    # Return list to be used in dropdown
    return l


def file_location_info(file):
    """
    Import a gpx file
    :param file: filename
    :return: location info
    """

    name, extension = os.path.splitext(file)
    if extension == ".gpx":
        gpx = gpxpy.parse(open(file, 'r'))
        lon = gpx.tracks[0].segments[0].points[0].longitude
        lat = gpx.tracks[0].segments[0].points[0].latitude

        geolocator = Nominatim(user_agent="geoapiExercises")
        location = geolocator.reverse(str(lat) + "," + str(lon))

        # return location.raw["display_name"]
        interesting_keys = ('suburb', 'village', 'municipality', 'county', 'state')
        s = ""
        first = True
        for key in interesting_keys:
            if key in location.raw['address']:
                if first:
                    s = s + "(" + location.raw['address'][key]
                    first = False
                else:
                    s = s + ", " + location.raw['address'][key]
        s = s + ")"
        return s

    else:
        return ""


def location_info(lon, lat, info='road'):
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.reverse(str(lat) + "," + str(lon))
    try:
        return location.raw['address'][info]
    finally:
        return None


def read_mapbox_token(file="mapbox_token.txt"):
    try:
        with open(file) as f:
            lines = f.readlines()
            return lines[0]
    except:
        return ""


mapbox_token = read_mapbox_token()


def import_gpx(file):
    """
    Import a gpx file
    :param file: filename
    :return: Pandas Dataframe
    """
    # Replacement value
    default_atemp = 26  # np.nan

    gpx = gpxpy.parse(open(file, 'r'))
    points = gpx.tracks[0].segments[0].points
    local_df = pd.DataFrame(columns=['time', 'lon', 'lat', 'elev', 'atemp', 'hr', 'cad'])

    # Base time if needed
    tz = datetime.timezone(datetime.timedelta(0))
    basetime = datetime.datetime(1900, 1, 1, 0, 0, 0, tzinfo=tz)

    for count, point in enumerate(
            tqdm(points, desc="Importing gpx \"{}\"".format(os.path.basename(file)[:6]), ncols=80)):
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
        local_df = pd.concat([local_df, pd.DataFrame(
            data={'time': point.time if point.time is not None else basetime + datetime.timedelta(seconds=count),
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
    # Replacement value
    default_atemp = 26  # np.nan

    with fitdecode.FitReader(file) as fit:

        local_df = pd.DataFrame(
            columns=['time', 'lon', 'lat', 'dist', 'elev', 'enhanced_elev', 'speed', 'enhanced_speed', 'atemp',
                     'hr', 'cad', 'fractional_cad'])

        for frame in tqdm(fit, desc="Importing fit \"{}\"".format(os.path.basename(file)[:6]), ncols=80):
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
                    try:
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
                    except:
                        pass

    return local_df


def load(files, debug_plots=False, debug=False, csv=False):
    """
    The object can load multiple gpx files at once. Useful when we want to plot multiple traces on the same map. The
    processing is done independently using local variables and the results are then appended to a class variable.
    All speeds are in m/s unless km/h are specified (and that can happen only in plots).
    If multiple files are loaded, an array of Dataframes is created.
    :param files: an array of .gpx files
    :param debug_plots: activates debug plots
    :param debug: verbose
    :param csv: dumps the df dataframe into a csv
    """

    # Create empty lists to be populated
    df_list = []  # "raw" traces
    df_moving_list = []  # traces with idle segments removed
    file_list = []

    # Go through all files
    for file in files:
        if debug:
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
        # Note: i'll be working from the gps_points object directly and then appending results into the dataframe. It
        # would make a lot more sense to operate directly on the dataframe.

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

        for idx in tqdm(range(1, len(df)), desc="Pre-processing \"{}\"".format(os.path.basename(file)[:6]), ncols=80):
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

        if debug_plots:
            fig_3 = px.line(df, x='time', y='c_dist_geo3d', template='plotly_dark')
            fig_3.show()

        # Speed
        df['c_speed'] = df['c_delta_geo3d'] / df['c_delta_time']

        # Remove absurd outliers
        df = df[df['c_speed'] < 80.0 / 3.6]

        # Check and fill nan's
        if debug:
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
        df_list.append(df)
        df_moving_list.append(df_moving)
        file_list.append(file)

        # Save to CSV for further processing (it's going to be faster even than just reimporting a GPX)
        if csv:
            df.to_csv("{}.csv".format(name))

    return df_list, df_moving_list, file_list


def create_historical(folder, debug_limit=0):
    """
    Load all gpx and fit files present in a folder (a Strava export, for example), strips all information apart from
    latitude and longitude and stores the resulting single long dataframe in a feather file.
    :param folder: the import folder
    :param debug_limit: if > 0, load only the first debug_limit files in alphabetical order
    :return: nothing, it saves a feather file
    """

    # Create folder for all .gz files
    gz_folder = os.path.join(folder, "gz")
    if not os.path.exists(gz_folder):
        os.makedirs(gz_folder)

    # Check for .gz files, if found, expand and move the gz to that folder
    for file in sorted(os.listdir(folder)):
        name, extension = os.path.splitext(file)
        if extension == ".gz":
            with gzip.open(os.path.join(folder, file), 'rb') as f_in:
                with open(os.path.join(folder, name), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    shutil.move(os.path.join(folder, file), os.path.join(gz_folder, file))

    # Read all files and store lon and lat in a dataframe
    all_df = pd.DataFrame(columns=['lon', 'lat'])
    counter = 0
    files = sorted(os.listdir(folder))
    for file in tqdm(files, desc="Importing traces", ncols=80):
        name, extension = os.path.splitext(file)
        stripped_df = None
        if extension == ".gpx":
            stripped_df = import_gpx(os.path.join(folder, file))[['lon', 'lat']]
            counter = counter + 1
        elif extension == ".fit":
            stripped_df = import_fit(os.path.join(folder, file))[['lon', 'lat']]
            counter = counter + 1

        all_df = pd.concat([all_df, stripped_df], axis=0, join='outer', ignore_index=True)

        if debug_limit != 0:
            if counter == debug_limit:
                break

    # Save dataframe in feather file
    feather.write_feather(all_df, 'heatmap/historical')
    return


def plot_historical_heatmap(center_lon, center_lat, lon_span=2, lat_span=1, file="heatmap/historical"):
    """
    Plots a heatmap based on data contained in a dataframe in a feather file. Filtering points by lon/lat is necessary
    to load the map, otherwise it crashes.
    :param center_lon: center lon coordinate of the rectangular area plotted
    :param center_lat: center lat coordinate of the rectangular area plotted
    :param lon_span: lon span of the rectangular area plotted
    :param lat_span: lat span of the rectangular area plotted
    :param file: feather file that contains a dataframe with lon and lat
    :return:
    """
    all_df = feather.read_feather(file)

    filtered_all_df = all_df.loc[
        (all_df['lat'] > (center_lat - lat_span / 2)) & (all_df['lat'] < (center_lat + lat_span / 2)) & (
                all_df['lon'] > (center_lon - lon_span / 2)) & (all_df['lon'] < (center_lon + lon_span / 2))]

    # print(filtered_all_df)

    fig = px.density_mapbox(filtered_all_df, lat='lat', lon='lon', z=None,
                            radius=4,
                            opacity=0.8,
                            zoom=6,
                            center=dict(lat=np.mean(filtered_all_df["lat"]), lon=np.mean(filtered_all_df["lon"])),
                            mapbox_style='open-street-map')
    fig.show()


def stats(df, df_moving):
    datecheck = datetime.datetime.strptime(df.iloc[0]["time"], "%Y-%m-%dT%H:%M:%S.%fZ")
    if datecheck.year != 1900:
        s = '''Geodesic Distance 2D: **{c_dist_geo2d:.3f} km**
        
Geodesic Distance 3D: **{c_dist_geo3d:.3f} km**
        
Distance Correction (geodesic 3D-2D): **{delta_c_dist_sph:.0f} m**
        
Total Time: **{total_time}**
        
Elevation Gain: **{elev_gain:.0f}**
        
Elevation Loss: **{elev_loss:.0f}**
        
Maximum Speed: **{max_c_speed:.1f} km/h**
        
Average Speed: **{avg_c_speed:.1f} km/h**
        
Average Moving Speed: **{avg_moving_c_speed:.1f} km/h**
        
Moving Time: **{moving_time}**
'''.format(c_dist_geo2d=df['c_dist_geo2d'].iloc[-1] / 1000,
           c_dist_geo3d=df['c_dist_geo3d'].iloc[-1] / 1000,
           delta_c_dist_sph=df['c_dist_geo3d'].iloc[-1] - df['c_dist_geo2d'].iloc[-1],
           total_time=str(datetime.timedelta(seconds=sum(df['c_delta_time']))),
           elev_gain=round(sum(df[df['c_delta_elev'] > 0]['c_delta_elev']), 2),
           elev_loss=round(sum(df[df['c_delta_elev'] < 0]['c_delta_elev']), 2),
           max_c_speed=round((3.6 * df['c_speed'].max(axis=0)), 2),
           avg_c_speed=round((3.6 * sum((df['c_speed'] * df['c_delta_time'])) / sum(df['c_delta_time'])), 2),
           avg_moving_c_speed=round(
               (3.6 * sum((df_moving['c_speed'] * df_moving['c_delta_time'])) / sum(df_moving['c_delta_time'])), 2),
           moving_time=str(datetime.timedelta(seconds=sum(df_moving['c_delta_time']))))
    else:
        s = '''Geodesic Distance 2D: **{c_dist_geo2d:.3f} km**

Geodesic Distance 3D: **{c_dist_geo3d:.3f} km**

Distance Correction (geodesic 3D-2D): **{delta_c_dist_sph:.0f} m**

Elevation Gain: **{elev_gain:.0f}**

Elevation Loss: **{elev_loss:.0f}**

**Timestamp missing in source file**
'''.format(c_dist_geo2d=df['c_dist_geo2d'].iloc[-1] / 1000,
           c_dist_geo3d=df['c_dist_geo3d'].iloc[-1] / 1000,
           delta_c_dist_sph=df['c_dist_geo3d'].iloc[-1] - df['c_dist_geo2d'].iloc[-1],
           elev_gain=round(sum(df[df['c_delta_elev'] > 0]['c_delta_elev']), 2),
           elev_loss=round(sum(df[df['c_delta_elev'] < 0]['c_delta_elev']), 2))

    return s


def plot_maps(df_list, file_list):
    """
    Plot all maps
    :return: figure
    """
    fig = go.Figure()
    for i, df in enumerate(df_list):
        fig.add_trace(go.Scattermapbox(lat=df["lat"],
                                       lon=df["lon"],
                                       mode='lines+markers',
                                       marker=go.scattermapbox.Marker(size=6),
                                       name=file_list[i],
                                       hovertext=df['c_dist_geo2d'],
                                       subplot='mapbox',
                                       )
                      )
        fig.update_layout(
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

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


def copy_segment(df, columns, interval_unit="m", interval=None):
    """
    Return a portion of one dataframe
    :param df: dataframe to operate on
    :param columns: columns that will be copied, it makes no sense to copy all because that will complicate
    operations such as interpolate unnecessarily
    :param interval_unit: can be "m" for meters or "i" for index
    :param interval: extremes (included) of the segment, in m
    :return:
    """
    # Select only the points belonging to the climb
    assert interval_unit in ("m", "i")

    # Create copy to avoid warnings and confusion
    copy_df = df[columns].copy()

    if interval is not None:
        # Selection in case of meter interval (used manually, need for "loop" behavior)
        if interval_unit == "m":
            max_dist = df['c_dist_geo2d'].max()
            min_dist = df['c_dist_geo2d'].min()
            start_dist = interval[0]
            if interval[1] == 0:
                end_dist = max_dist
            else:
                end_dist = interval[1]

            result_df = copy_df[(copy_df['c_dist_geo2d'] >= start_dist) & (copy_df['c_dist_geo2d'] <= end_dist)]

        # Selection in case of index interval (used by geppetto and based on selection, no need for "loop" behavior)
        elif interval_unit == "i":
            result_df = copy_df.iloc[interval[0]:interval[1]]

        return result_df

    else:
        # Copy the whole segment
        return copy_df


def plot_map(df, map_trace_color_param='elev', interval_unit="m", interval=None, hover_index=None, zoom=None):
    """

    :param df: dataframe to operate on
    :param map_trace_color_param: parameter to control the trace color
    :param interval_unit: 'm'=meters or 'i'=index
    :param interval: the interval of interest
    :param hover_index:
    :param zoom:
    :return:
    """

    df_selection = copy_segment(df,
                                columns=["lon", "lat", "c_dist_geo2d", "elev", 'c_speed'],
                                interval_unit=interval_unit,
                                interval=interval)
    df_not_selection = df[~df.index.isin(df_selection.index)]

    # Check that the desired highlight field exists
    if map_trace_color_param not in df_selection.columns:
        map_trace_color_param = 'elev'

    data = [
        go.Scattermapbox(lat=df_selection["lat"],
                         lon=df_selection["lon"],
                         mode='lines+markers',
                         line=dict(
                             width=2,
                             color="gray",
                         ),
                         marker=go.scattermapbox.Marker(size=6,
                                                        color=df_selection[map_trace_color_param],
                                                        colorscale=px.colors.sequential.Bluered),
                         hovertext=df_selection['c_dist_geo2d'],
                         subplot='mapbox2',
                         name='',
                         showlegend=False,
                         ),
        go.Scattermapbox(lat=df_not_selection["lat"],
                         lon=df_not_selection["lon"],
                         mode='markers',
                         marker=go.scattermapbox.Marker(size=6,
                                                        color='gray'),
                         hovertext=df_selection['c_dist_geo2d'],
                         subplot='mapbox2',
                         name='',
                         showlegend=False,
                         ),
    ]
    layout = go.Layout(hovermode='closest',
                       mapbox2=dict(style="open-street-map",
                                    accesstoken=mapbox_token,
                                    bearing=0,
                                    pitch=0,
                                    zoom=10,
                                    center=go.layout.mapbox.Center(lat=np.mean(df_selection["lat"]),
                                                                   lon=np.mean(df_selection["lon"])),

                                    ),
                       margin={'l': 0, 'b': 0, 't': 0, 'r': 0},
                       minreducedheight=400,
                       minreducedwidth=400,
                       paper_bgcolor='rgba(0,0,0,0)',
                       # autosize=False,
                       # uirevision=df_selection,
                       )
    fig = go.Figure(data=data, layout=layout)

    # Yellow dot
    if hover_index is not None:
        if (hover_index >= 0) and (hover_index <= df.shape[0]):
            fig.add_trace(go.Scattermapbox(lat=[df.iloc[hover_index]['lat']],
                                           lon=[df.iloc[hover_index]['lon']],
                                           mode='markers',
                                           marker=dict(
                                               size=6,
                                               color="yellow"),
                                           subplot='mapbox2',
                                           name='',
                                           showlegend=False,
                                           )
                          )

    # CONTROLLARE STO COSO
    # print(zoom)
    """
    if zoom:
        for axis_name in ['axis', 'axis2']:
            if f'x{axis_name}.range[0]' in zoom:
                fig['layout'][f'x{axis_name}']['range'] = [
                    zoom[f'x{axis_name}.range[0]'],
                    zoom[f'x{axis_name}.range[1]']
                ]
            if f'y{axis_name}.range[0]' in zoom:
                fig['layout'][f'y{axis_name}']['range'] = [
                    zoom[f'y{axis_name}.range[0]'],
                    zoom[f'y{axis_name}.range[1]']
                ]
    """

    return fig


def plot_elevation(df, hover_index=None):
    """
    Plots all traces that were imported.
    :return: Plots
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["c_dist_geo2d"],
                             y=df["elev"],
                             mode='lines+markers',
                             line=dict(
                                 width=1,
                                 color="red"),
                             marker=dict(
                                 size=1,
                                 color="red")
                             ),
                  )

    # Yellow dot
    if hover_index is not None:
        assert hover_index >= 0
        fig.add_trace(go.Scatter(x=[df.iloc[hover_index]['c_dist_geo2d']],
                                 y=[df.iloc[hover_index]['elev']],
                                 mode='markers',
                                 name="",
                                 marker=dict(
                                     size=4,
                                     color="yellow"),
                                 )
                      )

    fig.update_xaxes(showgrid=True,
                     showspikes=True,
                     title="2D distance (m)")
    fig.update_yaxes(showgrid=True,
                     title="Altitude (m)")
    fig.add_annotation(x=0.02, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text="Elevation")
    fig.update_layout(height=300,
                      margin={'l': 0, 'b': 0, 'r': 0, 't': 0},
                      hovermode='x',
                      selectdirection='h',
                      font=dict(
                          family="Helvetica",
                          size=10,
                          color="Gray",
                      ),
                      paper_bgcolor='rgba(0,0,0,0)',
                      )

    return fig


def gradient(df, interval_unit="m", interval=None, resolution=1000, slope_unit="per", show_map=False):
    """
    Computes the gradient over a portion of one dataframe
    :param df: dataframe to operate on
    :param interval_unit: can be "m" for meters or "i" for index
    :param interval: the gradient is calculated over the portion [start_meter, end_meter] of the input trace
    :param resolution: the "step" in which the gradient is calculated/averaged, in meters
    :param slope_unit: show slope as percentage ("per") or angle ("deg")
    :param show_map: show a minimap in the bottom right corner
    :return: a figure
    """

    def gradient_colorscale(gradient_value, half_n=15):
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

        To add:
        Average gradient
        Steepest 100m
        Length
        Total ascent
        """
        a = np.arange((half_n - 0.5), -half_n, -1.0)
        red = np.append(half_n * [255], np.linspace(255, 0, (half_n + 1)))
        green = np.append(np.linspace(0, 255, (half_n + 1)), half_n * [255])
        blue = (2 * half_n + 1) * [0]
        bin_index = np.digitize(gradient_value * 100, a)
        return "rgb({},{},{})".format(int(red[bin_index]), int(green[bin_index]), int(blue[bin_index]))

    # function to convert to superscript
    def get_super(x):
        normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
        super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
        res = x.maketrans(''.join(normal), ''.join(super_s))
        return x.translate(res)

    # Create a local copy of the input array of dataframes containing only the points belonging to the portion,
    # usually a climb, of interest
    df_climb = copy_segment(df,
                            columns=["lon", "lat", "c_dist_geo2d", "elev", 'c_delta_elev'],
                            interval_unit=interval_unit,
                            interval=interval)

    # Total distance and average gradient
    horizontal_dist = df_climb["c_dist_geo2d"].iloc[-1] - df_climb["c_dist_geo2d"].iloc[0]
    vertical_dist = df_climb["elev"].iloc[-1] - df_climb["elev"].iloc[0]
    avg_gradient = vertical_dist / horizontal_dist

    # Elevation gain and loss
    elev_gain = round(sum(df_climb[df_climb['c_delta_elev'] > 0]['c_delta_elev']), 2)
    elev_loss = round(sum(df_climb[df_climb['c_delta_elev'] < 0]['c_delta_elev']), 2)

    # Steepest 100m (only if the selected portion is short enough)
    if horizontal_dist < 30000:

        df_100m = df_climb.copy()
        steepest_100m_list = []

        # Stop search before 100 from the end
        search_limit = df_100m["c_dist_geo2d"].iloc[-1] - 101

        # For each point, calculate the average gradient of the 100m in front of it
        for index, row in df_100m.iterrows():
            if row['c_dist_geo2d'] < search_limit:
                # Append item 100m after the current point and tag it "mock"
                df_100m = pd.concat([df_100m, pd.DataFrame(data={'c_dist_geo2d': row['c_dist_geo2d'] + 100,
                                                                 'mock': True}, index=[0])],
                                    axis=0, join='outer', ignore_index=True)

                # Interpolate its values
                df_100m = df_100m.sort_values(by='c_dist_geo2d')
                df_100m = df_100m.interpolate(method='linear', limit_direction='backward', limit=1)
                index_of_mock = df_100m.index[df_100m['mock'] == True]
                # print(df_100m)

                # Elevation gain between a point and its mock 100m further
                # print(df_100m.loc[index_of_mock]['elev'])
                # print(row['elev'])
                # print("{:.1f}m over {:.1f}m  ".format(float(df_100m.loc[index_of_mock]['elev'] - row['elev']), float(df_100m.loc[index_of_mock]['c_dist_geo2d'] - row['c_dist_geo2d'])))
                steepest_100m_list.append(float(df_100m.loc[index_of_mock]['elev']) - row['elev'])
                # Note: the math would be gain / 100 * 100 to have a %. It would be a waste of power.

                # Remove mock point
                df_100m.drop(index=index_of_mock, inplace=True)

        steepest_100m = np.max(steepest_100m_list)
    else:
        steepest_100m = 0

    # Prepare for gradient plot
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
    df_climb_gradient['c_gradient'] = df_climb_gradient['c_elev_delta'] / df_climb_gradient['c_dist_delta']

    # This columns is redundant but is useful to cross check that the filter worked well
    # df_climb_gradient['steps'] = np.flip(steps)

    # Generate figure
    fig = go.Figure()
    for i in range(len(df_climb_gradient) - 1):
        portion = df_climb[
            (df_climb['c_dist_geo2d_neg'] >= df_climb_gradient.iloc[i]["c_dist_geo2d_neg"]) & (
                    df_climb['c_dist_geo2d_neg'] <= df_climb_gradient.iloc[i + 1]["c_dist_geo2d_neg"])]
        g = df_climb_gradient['c_gradient'].iloc[i]
        angle = np.arctan(g) / np.pi * 180
        fig.add_trace(go.Scatter(x=portion['c_dist_geo2d_neg'],
                                 y=portion['elev'],
                                 fill='tozeroy',
                                 fillcolor=gradient_colorscale(g),
                                 mode='none',
                                 name='',
                                 showlegend=False),
                      )
        annotation_per = "{:.0f}".format(np.trunc(g * 100)) + "{}".format(get_super("{:.0f}".format(np.trunc(((g * 100) % 1) * 10))))
        annotation_deg = "{:.0f}".format(np.trunc(angle * 100)) + "{}".format(get_super("{:.0f}".format(np.trunc(((angle * 100) % 1) * 10))))
        fig.add_annotation(x=np.mean(portion['c_dist_geo2d_neg']), y=np.max(portion['elev']), yshift=10,
                           text=annotation_per if slope_unit == "per" else annotation_deg,
                           showarrow=False,
                           arrowhead=0)

    # Minimap
    if show_map:
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
                                       hovertext=df_climb["c_dist_geo2d_neg"],
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
                pitch=0,
                zoom=11,
                center=go.layout.mapbox.Center(
                    lat=np.mean(df_climb["lat"]),
                    lon=np.mean(df_climb["lon"])
                ),
            )
        )

    fig.update_xaxes(showgrid=True,
                     showspikes=True,
                     title="2D distance to destination (m)")

    fig.update_yaxes(showgrid=True,
                     showspikes=True,
                     title="Altitude (m)")

    # Annotations
    fig.add_annotation(x=0.01, y=0.95, text="Distance: {:.3f} km".format(horizontal_dist / 1000),
                       xanchor='left', yanchor='bottom', xref='paper', yref='paper', showarrow=False, align='left')
    fig.add_annotation(x=0.01, y=0.90, text="Gain/Loss: {:.0f}/{:.0f} m".format(elev_gain, elev_loss),
                       xanchor='left', yanchor='bottom', xref='paper', yref='paper', showarrow=False, align='left')
    fig.add_annotation(x=0.01, y=0.85, text="Average gradient: {:.1f} %".format(avg_gradient * 100),
                       xanchor='left', yanchor='bottom', xref='paper', yref='paper', showarrow=False, align='left')
    fig.add_annotation(x=0.01, y=0.80, text="Steepest 100m: {:.1f} %".format(steepest_100m),
                       xanchor='left', yanchor='bottom', xref='paper', yref='paper', showarrow=False, align='left')

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      hovermode='x',
                      font=dict(
                          family="Helvetica",
                          size=10,
                          color="Gray",
                      ),
                      paper_bgcolor='rgba(0,0,0,0)',
                      )
    return fig


def estimate_power(df,
                   df_moving,
                   interval_unit="m",
                   interval=None,
                   total_mass=76 + 1 + 1.5 + 8,
                   filter_window=4,
                   debug_plots=False):
    """
    Estimates the power over a portion of one dataframe
    :param df: dataframe to operate on
    :param df_moving: moving avg dataframe to operate on
    :param interval_unit: can be "m" for meters or "i" for index
    :param interval: [start_meter, end_meter]
    :param total_mass: bike + rider
    :param filter_window: size of the rolling average filter window. 0 = no filter.
    :param debug_plots: toggle intermediate debug plots
    :return: a figure
    """

    def power_air(speed, altitude, temperature_degc, CdA=0.28, losses_drivetrain=0.051):
        """
        :param speed: [m/s]
        :param altitude: [m]
        :param temperature_degc: [°C]
        :param CdA: Coefficient of drag * area, to be estimated based
        :param losses_drivetrain: pedaling and drivetrain efficiency
        :return: [W]
        """
        # Air density
        L = 0.0065  # approximate rate of decrease in temperature with elevation
        T0 = 298  # sea level standard temperature
        M = 0.02896  # molar mass of dry air
        Rs = 287.058  # specific gas constant
        temperature_kelvin = temperature_degc + 273.15
        p_exp = M * const.g / (const.R * L)
        p = const.atm * (1 - (L * altitude / T0)) ** p_exp
        rho = p / (Rs * temperature_kelvin)  # kg/m^3
        force_air = 0.5 * CdA * rho * speed ** 2
        return force_air * speed / (1.0 - losses_drivetrain)

    def power_roll(gradient, m, speed, Crr=0.00321, losses_drivetrain=0.051):
        """
        Estimate rolling resistance for the 2 tires (TBC)
        :param gradient: ratio
        :param m: [kg]
        :param speed: [m/s]
        :param Crr: 0.00321 for one Continental GP5000 @6.9bar
        :param losses_drivetrain: pedaling and drivetrain efficiency
        :return: [W]
        """
        force_roll = 2. * Crr * np.cos(np.arctan(gradient)) * m * const.g
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

    # Check if timestamp exists. If not, just return None.
    # TODO
    # datecheck = datetime.datetime.strptime(df.iloc[0]["time"], "%Y-%m-%dT%H:%M:%S.%fZ")
    # if datecheck.year == 1900:
    #    return None

    # Work on a portion of the track
    df_selection = copy_segment(df,
                                columns=['time', 'lon', 'lat', 'c_dist_geo2d', 'elev', 'c_speed', 'atemp', 'hr', 'cad',
                                         'c_delta_time'],
                                interval_unit=interval_unit,
                                interval=interval)
    df_moving_selection = copy_segment(df_moving,
                                       columns=['c_delta_time', 'c_speed', 'c_dist_geo2d'],
                                       interval_unit=interval_unit,
                                       interval=interval)

    # Compute gradient
    df_selection['c_elev_delta'] = df_selection.elev.diff().shift(-1)
    df_selection['c_dist_delta'] = df_selection.c_dist_geo2d.diff().shift(-1)
    df_selection['c_gradient'] = df_selection['c_elev_delta'] / df_selection['c_dist_delta']

    if debug_plots:
        fig_5 = px.line(df_selection, x='c_dist_geo2d', y=['c_gradient', 'c_elev_delta', 'c_dist_delta'],
                        template='plotly_dark')
        fig_5.show()

    # Filter
    if filter_window > 0:
        df_selection['c_speed'] = df_selection['c_speed'].rolling(filter_window, center=True).mean()
        df_selection['c_gradient'] = df_selection['c_gradient'].rolling(filter_window, center=True).mean()
        df_selection['cad'] = df_selection['cad'].rolling(filter_window, center=True).mean()

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

    # Calculate average cadence removing 0 cadence points
    df_selection_0cadence = df_selection[df_selection['cad'] > 0.0]

    # Generate figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}], [{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=df_selection["c_dist_geo2d"],
                             y=df_selection["c_power_grav"],
                             name="Gravity",
                             hoverinfo='x+y',
                             mode='lines',
                             line=dict(width=0.0, color='green'),
                             stackgroup='one'
                             ),
                  row=1,
                  col=1,
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df_selection["c_dist_geo2d"],
                             y=df_selection["c_power_air"],
                             name="Air",
                             hoverinfo='x+y',
                             mode='lines',
                             line=dict(width=0.0, color='lightblue'),
                             stackgroup='one'  # define stack group
                             ),
                  row=1,
                  col=1,
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df_selection["c_dist_geo2d"],
                             y=df_selection["c_power_roll"],
                             name="Roll",
                             hoverinfo='x+y',
                             mode='lines',
                             line=dict(width=2.0, color='black'),
                             stackgroup='one'
                             ),
                  row=1,
                  col=1,
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df_selection["c_dist_geo2d"],
                             y=df_selection["hr"],
                             name="HR",
                             hoverinfo='x+y',
                             mode='lines',
                             line=dict(width=1.0, color='red'),
                             ),
                  row=1,
                  col=1,
                  secondary_y=True)

    fig.add_trace(go.Scatter(x=df_selection["c_dist_geo2d"],
                             y=3.6 * df_selection["c_speed"],
                             name="Speed",
                             hoverinfo='x+y',
                             mode='markers',
                             line=dict(width=1.0, color='blue'),
                             marker=dict(size=4, color="blue")
                             ),
                  row=2,
                  col=1,
                  secondary_y=False)

    fig.add_trace(go.Scatter(x=df_selection["c_dist_geo2d"],
                             y=df_selection["cad"],
                             name="Cadence",
                             hoverinfo='x+y',
                             mode='markers',
                             line=dict(width=1.0, color='grey'),
                             marker=dict(size=4, color="grey")
                             ),
                  row=2,
                  col=1,
                  secondary_y=True)

    fig.update_yaxes(row=1,
                     col=1,
                     secondary_y=False,
                     showgrid=True,
                     showspikes=True,
                     title="Power (W)")

    fig.update_yaxes(row=1,
                     col=1,
                     secondary_y=True,
                     showgrid=True,
                     showspikes=True,
                     title="HR (bpm)")

    fig.update_yaxes(row=2,
                     col=1,
                     secondary_y=False,
                     showgrid=True,
                     showspikes=True,
                     title="Speed (km/h)")

    fig.update_yaxes(row=2,
                     col=1,
                     secondary_y=True,
                     showgrid=True,
                     showspikes=True,
                     title="Cadence (rpm)")

    fig.update_xaxes(row=2,
                     col=1,
                     showgrid=True,
                     showspikes=True,
                     title="2D distance (m)")

    # Power annotation
    fig.add_annotation(x=0.00, y=0.50, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text="Average power: {:.0f} W".format(np.mean(df_selection['c_power'])))

    # Speed and cadence annotation
    annotation_text = """Average speed: {:.1f} km/h
        Average moving speed: {:.1f} km/h
        Average cadence: {:.0f} rpm
        Average pedaling cadence: {:.0f} rpm 
        """.format(
        round((3.6 * sum((df_selection['c_speed'] * df_selection['c_delta_time'])) / sum(df_selection['c_delta_time'])),
              2),
        round((3.6 * sum((df_moving_selection['c_speed'] * df_moving_selection['c_delta_time'])) / sum(
            df_moving_selection['c_delta_time'])), 2),
        np.mean(df_selection['cad']),
        np.mean(df_selection_0cadence['cad']))

    fig.add_annotation(x=0.00, y=0.45, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=annotation_text)

    fig.update_layout(legend=dict(orientation="v",
                                  yanchor="top",
                                  y=1,
                                  xanchor="left",
                                  x=1.02,
                                  ),
                      margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      hovermode='x',
                      height=600,
                      font=dict(
                          family="Helvetica",
                          size=10,
                          color="Gray",
                      ),
                      paper_bgcolor='rgba(0,0,0,0)',
                      )
    return fig


def cadence_speed_curve(df,
                        interval_unit="m",
                        interval=None):
    """
    This method operates on only one trace
    :param interval_unit:
    :param df: dataframe to operate on
    :param interval: [start_meter, end_meter]
    :return: plots
    """

    # Work on a portion of the track
    df_selection = copy_segment(df,
                                columns=['lon', 'lat', 'c_dist_geo2d', 'elev', 'cad', 'c_speed', 'hr'],
                                interval_unit=interval_unit,
                                interval=interval)

    cadence = np.linspace(0, 110, 100)
    gears = [50. / 11., 50. / 12., 50. / 13, 50. / 14., 50. / 16., 50. / 18, 50. / 20., 50. / 22., 50. / 25,
             50. / 28., 50. / 32., 34. / 11., 34. / 12., 34. / 13, 34. / 14., 34. / 16., 34. / 18, 34. / 20.,
             34. / 22., 34. / 25, 34. / 28., 34. / 32.]

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_selection["cad"],
                             y=df_selection["c_speed"] * 3.6,
                             mode='markers',
                             name="Measured",
                             marker=go.scatter.Marker(size=4,
                                                      opacity=0.1,
                                                      color=df_selection["hr"],
                                                      colorscale=px.colors.sequential.Redor),
                             )
                  )
    for gear in gears:
        speed = 2.0 * np.pi * (622.0 / 1000 / 2.0) * gear * (cadence / 60.0) * 3.6  # km/h
        fig.add_trace(go.Scatter(x=cadence,
                                 y=speed,
                                 mode='lines',
                                 name="{:.1f}".format(gear),
                                 line=dict(
                                     width=1,
                                     color="rgb(204,204,204)"),
                                 )
                      )
    fig.update_xaxes(range=[40, 110])
    fig.update_yaxes(range=[0, 60.])
    fig.update_layout(title="Cadence - Speed curve")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


def combine(folder, elements):
    """
    Combine intervals of a list of tracks into one single track
    :param list_of_tracks: list of (file, interval)
    :return:
    """

    # Read all files and store lon and lat in a dataframe
    all_df = pd.DataFrame(columns=['lon', 'lat', 'elev', 'c_dist_geo2d'])
    for element in elements:
        print("Importing \"{}\" from {} to {}m".format(element['file'], element['interval'][0], element['interval'][1]))
        df_list, _, _ = load([os.path.join(folder, element['file'])])
        df = df_list[0]
        df_selection = copy_segment(df,
                                    columns=['lon', 'lat', 'elev', 'c_dist_geo2d'],
                                    interval=element['interval'])

        all_df = pd.concat([all_df, df_selection], axis=0, join='outer', ignore_index=True)

    # Save into a gpx
    gpx = gpxpy.gpx.GPX()

    # Create first track in our GPX:
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)

    # Create first segment in our GPX track:
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    # Create points:
    for index, row in all_df.iterrows():
        gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(latitude=row['lat'],
                                                          longitude=row['lon'],
                                                          elevation=row['elev']))

    # Save file
    with open(os.path.join(folder, 'combined.gpx'), 'w') as f:
        f.write(gpx.to_xml())


def main():
    """
    Main function, used to test individual functions
    :return: nothing
    """

    if 0:
        df_list, df_moving_list, file_list = load(["tracks/The_missing_pass_W3_D2_.gpx",
                                                   "tracks/Local_passes_gravel_edition_.gpx",
                                                   "tracks/Two_more_W20_D3_.gpx",
                                                   "tracks/The_local_4_or_5_passes.gpx",
                                                   "tracks/More_local_passes_W17_D3_.gpx",
                                                   "tracks/More_local_4_passes.gpx",
                                                   "tracks/More_and_more_local_passes_W19_D3_.gpx",
                                                   "tracks/Even_more_local_passes.gpx",
                                                   "tracks/Cisa_e_Cirone.gpx",
                                                   "tracks/Autumnal_chestnut_trees_Cisa_and_Brattello.gpx",
                                                   ])
        plot_maps(df_list, file_list).show()

    if 0:
        # df_list, df_moving_list, file_list = load(["tracks/Caio 21.gpx"])
        df_list, df_moving_list, file_list = load(["tracks/The_missing_pass_W3_D2_.gpx"])

        df = df_list[0]
        df_moving = df_moving_list[0]

        # print(stats(df, df_moving))
        # plot_map(df).show()
        # plot_elevation(df).show()
        # gradient(df, interval=[33739, 34000], resolution=500).show()
        # estimate_power(df, df_moving, interval=None).show()
        # speed_cadence_timeseries(df, interval=[0, 0]).show()
        # cadence_speed_curve(df, interval=[0, 0]).show()

    # read_mapbox_token()

    if 0:
        # create_historical("/Users/ste/Downloads/export_19724628/activities/", debug_limit=0)
        plot_historical_heatmap(center_lon=9.5, center_lat=44.0)

    if 0:
        prova1 = (
            {'file': "UltraK_Trail_104km_6200m_d_9th_of_48_starters_25_finishers.gpx",
             'interval': (4200, 11100)},
            {'file': "Caio 21.gpx",
             'interval': (5800, 0)},
            {'file': "Caio 21.gpx",
             'interval': (0, 7400)},
            {'file': "UltraK_Trail_104km_6200m_d_9th_of_48_starters_25_finishers.gpx",
             'interval': (19700, 24800)},
        )
        prova2 = (
            {'file': "Caio 21.gpx",
             'interval': (5800, 0)},
            {'file': "Caio 21.gpx",
             'interval': (0, 7400)},
        )
        combine("/Users/ste/Library/CloudStorage/Dropbox/Cose divertenti/Sport/Trail/Parma/",
                prova2
                )


if __name__ == "__main__":
    main()
