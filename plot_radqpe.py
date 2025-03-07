#!/bin/env python3

# Load the pysteps library to use the importer functions from pysteps.io
import pysteps
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Parse the timestamp to be plotted from the input arguments
import os
import sys
import datetime

# Load the command line arguments
arg_list = sys.argv
if len(arg_list) < 2:
    print("Usage: plot_radqpe.py YYYYMMDDHHMM")
    sys.exit(1)

# Parse the timestamp to be plotted from the input arguments
startdate = datetime.datetime.strptime(arg_list[1], "%Y%m%d%H%M")

# Read the radar details from the pysteps rcparams.
data_src_radar= "rmi"

# typically "%Y%m%d%H%M%S.rad.bhbjbwdnfa.comp.rate.qpe2"
fn_pattern = pysteps.rcparams.data_sources[data_src_radar]["fn_pattern"]

# typically "hdf5"
fn_ext = pysteps.rcparams.data_sources[data_src_radar]["fn_ext"]
importer_name = pysteps.rcparams.data_sources[data_src_radar]["importer"]
importer_kwargs = pysteps.rcparams.data_sources[data_src_radar]["importer_kwargs"]
timestep = pysteps.rcparams.data_sources[data_src_radar]["timestep"]

# Find the radar data file
root_path = pysteps.rcparams.data_sources[data_src_radar]["root_path"]
path_fmt = pysteps.rcparams.data_sources[data_src_radar]["path_fmt"]

print("Loading and preprocessing radar analysis...")
print(startdate, root_path, path_fmt, fn_pattern, fn_ext, timestep)
fn_radar = pysteps.io.find_by_date(
    date=startdate,
    root_path=root_path,
    path_fmt=path_fmt,
    fn_pattern=fn_pattern,
    fn_ext=fn_ext,
    timestep=timestep,
    num_prev_files=0,
    num_next_files=23,
)
# Plot the radar data
print(fn_radar)
def plot_radar_data(fn_radar):
    # Get the correct importer
    importer_radar = pysteps.io.get_method(importer_name, "importer")

    radar_data, _, metadata = pysteps.io.read_timeseries(
        inputfns=fn_radar, importer=importer_radar, legacy=False
    )

    # Create a plot with the map of Belgium in the background
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=4.9, central_latitude=50.8, standard_parallels=(49.5, 52)))

    ax.coastlines(resolution='10m', color='black', linewidth=1)
    ax.set_extent([2.5, 6.5, 49.5, 52], ccrs.LambertConformal())

    # Plot the radar data
    plt.imshow(radar_data[0], extent=(metadata['x1'], metadata['x2'], metadata['y1'], metadata['y2']), origin='lower',
               cmap='viridis')

    plt.title('Radar Data')
    plt.show()
    # Save the plot to a file with the date
    plt.savefig(f"radar_{filename}.png")

plot_radar_data(filename)
