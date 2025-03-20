#!/bin/env python3

"""
Plot two nowcasts next to each other and the radar observation.
Author: Michiel Van Ginderachter 25/02/2022
michiel.vanginderachter@meteo.be
"""
import numpy as np

"""
Scipt needs to be run as:
plot_nwc.py YYYYMMDDHHMM 
with 
YYYYMMDDHHMM : analysisdate of the forecast (also used in .nc filename)
eg.: plot_nwc.py 202101270600 
"""

import os
import datetime
import sys
import matplotlib.pyplot as plt
import imageio

arg_list = sys.argv
if len(arg_list) < 1:
    print("Usage: plot_nwc.py YYYYMMDDHHMM")
    sys.exit(1)

import pysteps
from pysteps.io import import_netcdf_pysteps

control = False

# 1. Load the command line arguments
startdate = datetime.datetime.strptime(arg_list[1], "%Y%m%d%H%M")
threshold = 0.1
ncascade = 6

dir_base = "data" # change me
dir_nwc = os.path.join(dir_base, "nwc")
if control:
    dir_nwc_control = os.path.join(dir_base, "nwc_control")
dir_figs = os.path.join(dir_base, "figs_%s" % startdate.strftime("%Y%m%d%H%M"))
os.makedirs(dir_figs,exist_ok=True)

# Set the directories and data sources
data_src_radar = "rmi"

root_path = pysteps.rcparams.data_sources[data_src_radar]["root_path"]
path_fmt = pysteps.rcparams.data_sources[data_src_radar]["path_fmt"]
# BEWARE! This is not fixed in time. More recent radqpe files may have a different filename pattern.
fn_pattern = pysteps.rcparams.data_sources[data_src_radar]["fn_pattern"]
fn_ext = pysteps.rcparams.data_sources[data_src_radar]["fn_ext"]
importer_name = pysteps.rcparams.data_sources[data_src_radar]["importer"]
importer_kwargs = pysteps.rcparams.data_sources[data_src_radar]["importer_kwargs"]
timestep = pysteps.rcparams.data_sources[data_src_radar]["timestep"]

# Load the original nowcast
r_nwc, metadata = import_netcdf_pysteps(
    os.path.join(dir_nwc, "blended_nowcast_%s.nc" % startdate.strftime("%Y%m%d%H%M"))
)

if control :
    # Load the control nowcast
    r_nwc_control, metadata = import_netcdf_pysteps(
        os.path.join(dir_nwc_control, "blended_nowcast_%s.nc" % startdate.strftime("%Y%m%d%H%M"))
    )

# The shape of the ensemble nowcast is (n_ens, n_lt, x, y) but the control nowcast and the ensemble nowcast
# with 1 ensemble member is (n_lt, x, y). Determine the min common lead time of the two nowcasts.
n_lt = r_nwc.shape[1] if len(r_nwc.shape) == 4 else r_nwc.shape[0]
if control:
    n_lt = min(n_lt, r_nwc_control.shape[0])


print("Loading and preprocessing radar analysis...")
fn_radar = pysteps.io.find_by_date(
    date=startdate,
    root_path=root_path,
    path_fmt=path_fmt,
    fn_pattern=fn_pattern,
    fn_ext=fn_ext,
    timestep=timestep,
    num_prev_files=0,
    num_next_files=n_lt-1,
)

importer_radar = pysteps.io.get_method(importer_name, "importer")
r_radar, _, metadata_radar = pysteps.io.read_timeseries(
    inputfns=fn_radar, importer=importer_radar, legacy=False
)

coord_x = r_nwc.shape[2] * 2 // 3
coord_y = r_nwc.shape[3] * 1 // 3
half_box_width = 50
def spaghetti_plot(nowcast, nowcast_control, analysisdate, t_step, dir_figs):
    """
    Create a spaghetti plot of the time evolution of the nowcast precipitation intensity for a center 100x100 box in the domain.
    The observed radar rainfall is added in another colour. The plot is saved in the directory dir_gif.
    """
    # Sum over 100x100 center box
    nowcast_sum = np.nanmean(nowcast[:, :, coord_x - half_box_width : coord_x + half_box_width, coord_y - half_box_width : coord_y + half_box_width], axis=(2, 3))
    nowcast_control_sum = np.nanmean(nowcast_control[:, coord_x - half_box_width : coord_x + half_box_width, coord_y - half_box_width : coord_y + half_box_width], axis=(1, 2))
    r_radar_sum = np.nanmean(r_radar[:, coord_x - half_box_width : coord_x + half_box_width, coord_y - half_box_width : coord_y + half_box_width], axis=(1, 2))

    plt.figure(figsize=(8, 6))
    for j in range(nowcast.shape[0]):
        plt.plot(
            np.arange(1, nowcast.shape[1] + 1) * t_step,
            nowcast_sum[j, :],
            # nowcast[j, :, coord_x, coord_y],
            label="Member %i" % j,
            alpha=0.5,
        )
    # plot the observations as a thicker line
    plt.plot(
        np.arange(1, nowcast.shape[1] + 1) * t_step,
        r_radar_sum[:],
        # r_radar[:,  coord_x, coord_y],
        label="Observations",
        color="black",
        linewidth=2,
    )
    # plot the control nowcast as a dashed line
    plt.plot(
        np.arange(1, nowcast.shape[1] + 1) * t_step,
        nowcast_control_sum[:],
        # nowcast_control[:, coord_x, coord_y],
        label="Control nowcast",
        color="red",
        linestyle="--",
    )

    # plt.legend()
    plt.xlabel("Forecast lead time (min)")
    plt.ylabel("Precipitation intensity (mm/h)")
    plt.title(
        "Spaghetti plot for the precipitation ensemble nowcast\n"
        "validdate: %s" % analysisdate.strftime("%Y-%m-%d %H:%M")
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            dir_figs,
            "forecast_%s_spaghetti.png"
            % (analysisdate.strftime("%Y%m%d%H%M")),
        ),
        dpi=72,
    )
    print("Saved spaghetti plot as %s" % os.path.join(dir_figs, "forecast_%s_spaghetti.png" % (analysisdate.strftime("%Y%m%d%H%M"))))
    plt.close()


#

# plot the min-max forecast range as a semi-transparent band, and plot the radar observations as a darker line.
def uncertainty_plot(nowcast, nowcast_control, analysisdate,t_step,dir_figs):
    plt.figure(figsize=(8,6))
    # Plot the min-max forecast range as a semi-transparent band
    plt.fill_between(np.arange(1, nowcast.shape[1] + 1) * t_step,
                        np.nanmin(nowcast[:, :, coord_x, coord_y], axis=0),
                        np.nanmax(nowcast[:, :, coord_x, coord_y], axis=0),
                        color="blue", alpha=0.3, label="Min-max range")
    # Plot the radar observations as a darker line
    plt.plot(np.arange(1, nowcast.shape[1] + 1) * t_step,
                r_radar[:nowcast.shape[1], coord_x, coord_y],
                color="black", label="Radar observations")
    # Plot the control nowcast as a dashed line
    plt.plot(np.arange(1, nowcast.shape[1] + 1) * t_step,
                nowcast_control[:, coord_x, coord_y],
                color="red", linestyle="--", label="Control nowcast")

    plt.legend()
    plt.xlabel("Forecast lead time (min)")
    plt.ylabel("Precipitation intensity (mm/h)")
    plt.title("Uncertainty plot for the precipitation ensemble nowcast\n"
              "validdate: %s" % analysisdate.strftime("%Y-%m-%d %H:%M"))
    plt.tight_layout()
    plt.savefig(os.path.join(dir_figs, "forecast_%s_uncertainty.png" % (analysisdate.strftime("%Y%m%d%H%M"))), dpi=72)
    print("Saved uncertainty plot as forecast_%s_uncertainty.png" % (analysisdate.strftime("%Y%m%d%H%M")))
    plt.close()


def density_plot(nowcast,nowcast_control, analysisdate,t_step,dir_figs):
    plt.figure(figsize=(8,6))
    # Plot the forecast range of the ensemble with opacity proportional to the density of ensemble members
    for j in range(nowcast.shape[0]):
        plt.plot(np.arange(1, nowcast.shape[1] + 1) * t_step,
                    nowcast[j, :, coord_x, coord_y],
                    color="blue", alpha=0.2)
    # Plot the mean of the ensemble as a thicker line
    plt.plot(np.arange(1, nowcast.shape[1] + 1) * t_step,
                np.nanmean(nowcast[:, :, coord_x, coord_y], axis=0),
                color="red", linewidth=2, alpha=0.5, label="Ensemble mean")
    # Plot the ensemble median as a thicker, dashed line
    plt.plot(np.arange(1, nowcast.shape[1] + 1) * t_step,
                np.nanmedian(nowcast[:, :, coord_x, coord_y], axis=0),
                color="blue", linewidth=2, linestyle="--", label="Ensemble median")
    # Plot the control nowcast as a thin dashed line
    plt.plot(np.arange(1, nowcast.shape[1] + 1) * t_step,
                nowcast_control[:, coord_x, coord_y],
                color="red", linestyle="--", label="Control nowcast")

    # Plot the radar observations as a darker line
    plt.plot(np.arange(1, nowcast.shape[1] + 1) * t_step,
                r_radar[:nowcast.shape[1], coord_x, coord_y],
                color="black", label="Radar observations")
    plt.legend()
    plt.xlabel("Forecast lead time (min)")
    plt.ylabel("Precipitation intensity (mm/h)")
    plt.title("Density plot for the precipitation ensemble nowcast.\n"
              "validdate: %s" % analysisdate.strftime("%Y-%m-%d %H:%M"))
    plt.tight_layout()
    plt.savefig(os.path.join(dir_figs, "forecast_%s_density.png" % (analysisdate.strftime("%Y%m%d%H%M"))), dpi=72)
    print("Saved density plot as forecast_%s_density.png" % (analysisdate.strftime("%Y%m%d%H%M")))
    plt.close()

# plot a single ensemble member and the radar observations
def single_member_plot(nowcast, analysisdate,t_step,dir_figs):
    plt.figure(figsize=(8,6))
    # Plot a single ensemble member and the radar observations
    plt.plot(np.arange(1, nowcast.shape[1] + 1) * t_step,
                nowcast[0, :, coord_x, coord_y],
                color="blue", alpha=0.5, label="Ensemble member 1")
    plt.plot(np.arange(1, nowcast.shape[1] + 1) * t_step,
                r_radar[:nowcast.shape[1], coord_x, coord_y],
                color="black", label="Radar observations")
    plt.legend()
    plt.xlabel("Forecast lead time (min)")
    plt.ylabel("Precipitation intensity (mm/h)")
    plt.title("Single ensemble member plot for the precipitation ensemble nowcast.\n"
              "validdate: %s" % analysisdate.strftime("%Y-%m-%d %H:%M"))
    plt.tight_layout()
    plt.savefig(os.path.join(dir_figs, "forecast_%s_single_member.png" % (analysisdate.strftime("%Y%m%d%H%M"))), dpi=72)
    print("Saved single member plot as forecast_%s_single_member.png" % (analysisdate.strftime("%Y%m%d%H%M")))
    plt.close()

print(" Shape of r_nwc:",    r_nwc.shape)
if control :
    print(" Shape of r_nwc_control:",    r_nwc_control.shape)
print(" Shape of r_radar:", r_radar.shape)

if control :
    spaghetti_plot(r_nwc, r_nwc_control, startdate, timestep, dir_figs)
    uncertainty_plot(r_nwc, r_nwc_control,startdate, timestep, dir_figs)
    density_plot(r_nwc, r_nwc_control, startdate, timestep, dir_figs)

single_member_plot(r_nwc, startdate, timestep, dir_figs)

# Plot the map of the probability of precipitation exceeding a certain threshold thr at a given leadtime lt.
def plot_exceedance_probability(nowcast,thr,lt,analysisdate,dir_figs):
    plt.figure(figsize=(8,6))
    # Plot the exceedance probability map
    plt.imshow(np.sum(nowcast[:, lt, :, :] > thr, axis=0) / nowcast.shape[0],
                origin="upper", cmap="jet", vmin=0, vmax=1)
    plt.colorbar()
    plt.title("Probability of exceeding %g mm/h at lead time %i min\n" % (thr, lt))
    plt.tight_layout()
    fig_fn = "forecast_%s_exceedance_probability_thr_%s_lt_%d.png" % (analysisdate.strftime("%Y%m%d%H%M"), thr, lt)
    plt.savefig(os.path.join(dir_figs, fig_fn), dpi=72)
    print("Saved exceedance probability plot as %s" % fig_fn)
    plt.close()
    
# Plot the forecasts on a map.
def plot_nowcast(nowcast,metadata,product,member=None,pthr=None,figdir='./',startdate=None,geometries=None,dpi=72,height=1085,width=1029):
    '''
    Plotting:
        Plots all timesteps of an ensemble member, reusing the fig and ax for eacht timestep
        to increase speed.
    Inputs:
        metadata:   pysteps metadata dict
        product:    which product to plot {'member','prob', 'mean' or 'radar'}
        member:     which member we're plotting (only needed if prod = 'member')
        pthr:       threshold for probability (only needed if prod = 'prob')
        figdir:     directory where to store the figures (default: ./)
        startdate:  startdate of the nowcast (if None it is calculated from metadata)
        geometries:   dict with additional geometries and plotting arguments as kwargs
        figdir
    Outputs:
        list of filenames
    '''
    from pysteps.visualization.utils import get_geogrid, get_basemap_axis
    from pysteps.visualization.precipfields import get_colormap
    import os
    # set startdate if not provided
    if startdate is None:
        startdate = metadata['timestamps'][0] - dt.timedelta(minutes=metadata['leadtimes'][0])
    # set pixel-value
    px = 1./dpi
    # create list to save filenames
    filenames = []
    # set fixed titles
    if product == 'prob':
        producttitle = 'Probability of precipitation rate > %.1f mm/h' % (pthr,)
        data = np.sum(nowcast > pthr, axis=0) / nowcast.shape[0]
        figdir = os.path.join(figdir,'steps_pr%s' % "".join(str(pthr).split('.')))
        filebase = 'steps_pr%s_ac%02dA0' % (
            "".join(str(pthr).split('.')),
            metadata['accutime']
            )
        # colorbar settings
        cax = None
        extend = "neither"
        ptype = 'prob'
        units = metadata['unit']
        label_title = f"P(R > {pthr:.1f} {units})"
        cmap, norm, clevs, clevs_str = get_colormap(
            ptype=ptype,
            units=units,
            )
    else:
        # colorbar settings
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
        if product == 'member':
            producttitle = 'Ensemble Member %i' % (member)
            data = nowcast[member,:,:,:]
            figdir = os.path.join(figdir,'steps_en%02d' % (member,))
            filebase = 'steps_en%02d_ac%02dA0' %(member, metadata['accutime'])
        elif product == 'mean':
            producttitle = "Ensemble Mean"
            # Mean over the members of nowcast:
            data = np.nanmean(nowcast,axis=0)
            figdir = os.path.join(figdir,'steps_mean')
            filebase = 'steps_mean_ac%02dA0' % (metadata['accutime'], )
        elif product == 'control':
            producttitle = "Control Member"
            data = nowcast
            figdir = os.path.join(figdir, 'steps_control')
            filebase = 'steps_control_ac%02dA0' % (metadata['accutime'],)
        elif product == 'radar':
            producttitle = "Radar Observation"
            data = nowcast
            figdir = os.path.join(figdir, 'steps_radar')
            filebase = 'steps_radar_ac%02dA0' % (metadata['accutime'],)
        else:
            logger.error("Product %s not supported, quitting!" % (product,))
            sys.exit(1)

    os.makedirs(figdir,exist_ok=True)
    accutitle = 'Temporal Accumulation: %i min' % metadata['accutime']
    resotitle = 'Spatial Resolution: %.1f km' % (metadata['xpixelsize']/1000)
    nwptitle = "Blended nowcast"
    if "nwp_models" in metadata.keys():
        if product == 'member':
            splitter = int(nowcast.shape[0] / len(metadata['nwp_models']))
            imodel = int(member / splitter)
            nwptitle = 'Blended with %s - %s UTC' % (metadata['nwp_models'][imodel],
                    metadata['nwp_dates'][0].strftime('%Y-%m-%d %H:%M'))
        else:
            nwptitle = 'Blended with %s - %s UTC' % (", ".join(metadata['nwp_models']),
                    ", ".join([item.strftime('%Y-%m-%d %H:%M') for item in metadata['nwp_dates']]))
    # set grid specifications
    x_grid, y_grid, extent, regular_grid, origin = get_geogrid(
        nlon = data.shape[1], nlat = data.shape[2], geodata=metadata)
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
    ax.text(0,-0.03,nwptitle,transform=ax.transAxes,ha='left',size=10)
    # set timestep for first figure
    timestep=0
    # add the data
    im = ax.imshow(
        np.ma.masked_invalid(data[timestep,:,:]),
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
    suptitle = "%s + %i min" %  (
        startdate.strftime("%a %d %b %Y %H:%M:%S"),
        metadata['leadtimes'][timestep]
        )
    title = "%sZ nowcast" % (metadata['timestamps'][timestep].strftime('%H:%M'), )
    ax_title = ax.text(0,1.02,title,transform=ax.transAxes,fontdict= {'color' : 'purple', 'size' : 15})
    ax_suptitle = ax.text(0,1.06,suptitle,transform=ax.transAxes)
    ax.axis('off')
    # save figure and add filename to list
    filename='%s+%02d.png' % (filebase, timestep+1)
    filepath=os.path.join(figdir,filename)
    # This strange loop is needed to converge the figure margins to a stable value
    # This seems to be a matplotlib bug (TODO: check if a newer matplotlib-version fixes this.)
    for _ in range(30):
        fig.savefig(filepath, dpi=dpi)
    filenames.append(filepath)
    # start loop over remaining timesteps, reusing the figure and axis for speedup
    for timestep in range(1,data.shape[0]):
        im.set_data(np.ma.masked_invalid(data[timestep,:,:]))
        suptitle = "%s + %i min" %  (
            startdate.strftime("%a %d %b %Y %H:%M:%S"),
            metadata['leadtimes'][timestep]
            )
        title = "%sZ nowcast" % (metadata['timestamps'][timestep].strftime('%H:%M'), )
        ax_title.set_text(title)
        ax_suptitle.set_text(suptitle)
        ax.set_axis_off()
        filename='%s+%02d.png' % (filebase, timestep+1)
        filepath=os.path.join(figdir,filename)
        fig.savefig(filepath, dpi=dpi)
        filenames.append(filepath)

    print("Created %i figures in %s" % (len(filenames),figdir))
    print("Filenames: ",filenames)
    return(filenames)

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

plot_maps = True
if plot_maps:
    plot_nowcast(r_nwc,metadata, 'member', member=0, figdir=dir_figs, startdate=startdate)
    plot_nowcast(r_nwc,metadata, 'mean', figdir=dir_figs, startdate=startdate)
    if control :
        plot_nowcast(r_nwc_control,metadata, 'control', figdir=dir_figs, startdate=startdate)
    plot_nowcast(r_radar,metadata,'radar', figdir=dir_figs, startdate=startdate)
    plot_nowcast(r_nwc,metadata, 'prob', pthr=0.1, figdir=dir_figs, startdate=startdate)
    plot_nowcast(r_nwc, metadata, 'prob', pthr=5.0, figdir=dir_figs, startdate=startdate)

# end the program
sys.exit(0)

# Create a GIF of the time evolution of the nowcast precipitation intensity for the center pixel of the domain.

filenames = []
for i in range(r_nwc.shape[1]):
    title = "Precipitation nowcast %s + %i min\nvaliddate: %s" % (
        startdate.strftime("%Y-%m-%d %H:%M"),
        (i + 1) * timestep,
        startdate + datetime.timedelta(minutes=(i + 1) * 5),
    )
    plt.figure(figsize=(16, 12))
    for j in range(r_nwc.shape[0]):
        plt.subplot(6, 4, j + 1)
        pysteps.visualization.plot_precip_field(
            r_nwc[j, i, :, :], geodata=metadata_radar, title=r"Member %i" % j
        )
    plt.suptitle(title)
    plt.tight_layout()
    filename = f"{i}.png"
    filenames.append(filename)
    plt.savefig(filename, dpi=72)
    plt.close()

# build gif
kargs = {"duration": 0.4}
with imageio.get_writer(
        os.path.join(
            dir_figs,
            r"forecast_%s.gif"
            % (startdate.strftime("%Y%m%d%H%M")),#, date_nwp.strftime("%Y%m%d%H%M")),
        ),
        mode="I",
        **kargs,
) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# remove files
for filename in set(filenames):
    os.remove(filename)

