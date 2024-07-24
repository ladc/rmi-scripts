#!/bin/env python3

"""
Plot radar observations. Based on script by Michiel Van Ginderachter
michiel.vanginderachter@meteo.be
Run script without arguments for usage information.
"""
import numpy as np
import os
import datetime
import sys
import matplotlib.pyplot as plt

arg_list = sys.argv
if len(arg_list) < 1:
    print("Usage: plot_obs.py YYYYMMDDHHMM [NFILES]")
    sys.exit(1)

import pysteps

# 1. Load the command line arguments
startdate = datetime.datetime.strptime(arg_list[1], "%Y%m%d%H%M")
nfiles = int(arg_list[2]) if len(arg_list) > 2 else 1
threshold = 0.1

dir_base = ".."  # change me
dir_figs = os.path.join(dir_base, "figs")
# Set the directories and data sources
data_src_radar = "rmi"

# Hard-coding some paths here to avoid potential pystepsrc issues.
root_path = os.path.join(dir_base,'hackathon_testdata/radar') # pysteps.rcparams.data_sources[data_src_radar]["root_path"]
path_fmt = f'%Y%m%d' #pysteps.rcparams.data_sources[data_src_radar]["path_fmt"]
# BEWARE! This is not fixed in time. More recent radqpe files may have a different filename pattern.
fn_pattern = '%Y%m%d%H%M%S.rad.best.comp.rate.qpe' #pysteps.rcparams.data_sources[data_src_radar]["fn_pattern"]
fn_ext = 'hdf' #pysteps.rcparams.data_sources[data_src_radar]["fn_ext"]
importer_name = pysteps.rcparams.data_sources[data_src_radar]["importer"]
importer_kwargs = pysteps.rcparams.data_sources[data_src_radar]["importer_kwargs"]
timestep = pysteps.rcparams.data_sources[data_src_radar]["timestep"]


print("Loading and preprocessing radar analysis...")
fn_radar = pysteps.io.find_by_date(
    date=startdate,
    root_path=root_path,
    path_fmt=path_fmt,
    fn_pattern=fn_pattern,
    fn_ext=fn_ext,
    timestep=timestep,
    num_prev_files=0,
    num_next_files=nfiles-1,
)

importer_radar = pysteps.io.get_method(importer_name, "importer")
r_radar, _, metadata = pysteps.io.read_timeseries(
    inputfns=fn_radar, importer=importer_radar, legacy=False
)
print(r_radar.shape)
print(metadata)

# Plot the forecasts on a map.
def plot_radar(radar,metadata,figdir='./',geometries=None,dpi=72,height=1085,width=1029):
    '''
    Plotting:
        Plots all timesteps of an ensemble member, reusing the fig and ax for eacht timestep
        to increase speed.
    Inputs:
        radar:      radar data array
        metadata:   pysteps metadata dict
        figdir:     directory where to store the figures (default: ./)
        geometries:   dict with additional geometries and plotting arguments as kwargs
    Outputs:
        list of filenames
    '''
    from pysteps.visualization.utils import get_geogrid, get_basemap_axis
    from pysteps.visualization.precipfields import get_colormap
    import os
    # set pixel-value
    px = 1./dpi
    # create list to save filenames
    filenames = []
    cax = None
    extend = "max"
    ptype = 'intensity'
    units = metadata['unit']
    label_title = f"Precipitation intensity [{units}]"
    cmap, norm, clevs, clevs_str = get_colormap(
        ptype=ptype,
        units=units,
        colorscale='STEPS-BE'
        )
    producttitle = "Radar Observation"
    figdir = os.path.join(figdir, 'observations')
    filebase = 'steps_radar'

    os.makedirs(figdir,exist_ok=True)
    accutitle = 'Temporal Accumulation: %i min' % metadata['accutime']
    resotitle = 'Spatial Resolution: %.1f km' % (metadata['xpixelsize']/1000)
   # set grid specifications
    x_grid, y_grid, extent, regular_grid, origin = get_geogrid(
        nlon = radar.shape[1], nlat = radar.shape[2], geodata=metadata)
    # create figure and axis and plot fixed elements (coastline, countries,...)
    fig, ax = plt.subplots(figsize=(height*px,width*px),dpi=dpi,layout="constrained")
    ax = get_basemap_axis(extent, ax=ax, geodata=metadata, map_kwargs={'scale' : '10m'})
    if geometries is not None:
        for geo in geometries:
            ax.add_geometries(
                geometries[geo]['shapes'],
                crs=ccrs.PlateCarree(),**geometries[geo]['kwargs']
                )
    ax.text(1,1.02,producttitle,transform=ax.transAxes,ha='right',color='r')
    ax.text(1,-0.02,accutitle,transform=ax.transAxes,ha='right',size=8)
    ax.text(1,-0.04,resotitle,transform=ax.transAxes,ha='right',size=8)
    # set timestep for first figure
    timestep=0
    timestamp = metadata['timestamps'][0]
    # add the data
    im = ax.imshow(
        np.ma.masked_invalid(radar[timestep,:,:]),
        cmap=cmap,
        norm=norm,
        extent=extent,
        interpolation='nearest',
        origin=origin,
        zorder=10
        )
    # add colorbar
    cbar = plt.colorbar(
        im, ticks=clevs, spacing="uniform", extend=extend, shrink=0.8, cax=cax
        )
    cbar.ax.set_yticklabels(clevs_str)
    cbar.set_label(label_title)
    # add title and supertitle
    suptitle = "%s " % (
        timestamp.strftime("%a %d %b %Y %H:%M:%S"),
    )
    title = "%sZ observation" % (metadata['timestamps'][timestep].strftime('%H:%M'), )
    ax_title = ax.text(0,1.02,title,transform=ax.transAxes,fontdict= {'color' : 'purple', 'size' : 15})
    ax_suptitle = ax.text(0,1.06,suptitle,transform=ax.transAxes)
    ax.axis('off')
    # save figure and add filename to list
    filename='%s_%s.png' % (filebase, timestamp.strftime("%Y%m%d%H%M"))
    filepath=os.path.join(figdir,filename)
    # This strange loop is needed to converge the figure margins to a stable value
    # This seems to be a matplotlib bug (TODO: check if a newer matplotlib-version fixes this.)
    for _ in range(30):
        fig.savefig(filepath, dpi=dpi)
    filenames.append(filepath)
    # start loop over remaining timesteps, reusing the figure and axis for speedup
    for timestep in range(1,r_radar.shape[0]):
        im.set_data(np.ma.masked_invalid(radar[timestep,:,:]))
        timestamp = metadata['timestamps'][timestep]
        suptitle = "%s " %  (
            timestamp.strftime("%a %d %b %Y %H:%M:%S"),
            )
        title = "%sZ nowcast" % (metadata['timestamps'][timestep].strftime('%H:%M'), )
        ax_title.set_text(title)
        ax_suptitle.set_text(suptitle)
        ax.set_axis_off()
        filename = '%s_%s.png' % (filebase, timestamp.strftime("%Y%m%d%H%M"))
        filepath=os.path.join(figdir,filename)
        fig.savefig(filepath, dpi=dpi)
        filenames.append(filepath)

    print("Created %i figures in %s" % (len(filenames),figdir))
    print("Filenames: ",filenames)
    return(filenames)

# Add some missing info to metadata
metadata['projection'] = '+proj=lcc +lat_1=49.83333333333334 +lat_2=51.16666666666666 '+ \
    '+lat_0=50.797815 +lon_0=4.359215833333333 '+ \
    '+x_0=349328.0 +y_0=-334738.0 '+ \
    '+ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
metadata['x1'] = 6.39486825093627e-06
metadata['y1'] = -699999.9998974936
metadata['x2'] = 699999.9999935811
metadata['y2'] = 9.94016882032156e-05
metadata["xpixelsize"] = 1000
metadata["ypixelsize"] = 1000
metadata["yorigin"] = "upper"

plot_radar(r_radar,metadata, figdir=dir_figs)
