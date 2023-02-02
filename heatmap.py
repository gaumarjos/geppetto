import argparse
import mate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Heatmap generator',
        description='Create a heatmap based on .gpx and .fit files contained in a folder (e.g. exported from Strava. As reading a number of activity files takes time, this app stores the imported files in a "historical" file. If you want to regenerate that file, i.e. import new activities, specify the folder where all activities are stored.')
    parser.add_argument('-f', '--folder', dest='folder', required=False, default=None,
                        help="Folder where the activity files are located. Use ONLY to reimport all activities.")
    parser.add_argument('--lonc', dest='lon_center', required=True,
                        help="Center longitude of the area displayed")
    parser.add_argument('--latc', dest='lat_center', required=True,
                        help="Center latitude of the area displayed")
    parser.add_argument('--lons', dest='lon_span', required=False, default=2,
                        help='Longitude span of the area displayed. 2° if not specified.')
    parser.add_argument('--lats', dest='lat_span', required=False, default=1,
                        help='Latitude span of the area displayed. 1° if not specified.')
    args = parser.parse_args()

    if args.folder is not None:
        mate.create_historical(args.folder)
    mate.plot_historical_heatmap(center_lon=float(args.lon_center),
                                 center_lat=float(args.lat_center),
                                 lon_span=float(args.lon_span),
                                 lat_span=float(args.lat_span))
