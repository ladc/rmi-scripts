#!/bin/env python3

"""
Run a stochastic nowcast blended with NWP forecast based on STEPS (Bowler et. al 2006)
This script assumes the NWP forecast cascade decomposition is already performed with
decompose_nwp.py
Author: Michiel Van Ginderachter 25/02/2022
michiel.vanginderachter@meteo.be
"""
"""
Scipt needs to be run as:
run_steps.py YYYYMMDDHHMM FC NENS CPUS
with 
YYYYMMDDHHMM : startdate of the forecast (also used in .nc filename)
FC           : forecast length (number of timesteps)
NENS         : number of ensemble members
CPUS         : number of workers
eg.: run_steps.py 202107041600 20 12 4
"""

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pysteps
import sys

arg_list = sys.argv
if len(arg_list) < 2:
    print("Usage: run_steps.py YYYYMMDDHHMM FC NENS CPUS")
    print("e.g. \n run_steps.py 202107041600 12 4 4")
    sys.exit(1)

import dask
import pysteps

# 1. Load the command line arguments
startdate = datetime.datetime.strptime(arg_list[1],"%Y%m%d%H%M")
fc_length = int(arg_list[2])
nens = int(arg_list[3])
ncores = int(arg_list[4])

# Default parameters
# Rain/No-rain threshold
threshold = 0.1 
# Number of cascade levels
ncascade = 6

# Directories
dir_base = "data" # change me - this is the dir that contains the hackathon_testdata directory and where the output will be written
dir_cascade = os.path.join(dir_base,'nwp')#,startdate.strftime('%Y%m%d'))
dir_motion = dir_cascade
dir_skill = os.path.join(dir_base,'skill')
dir_gif = os.path.join(dir_base,'gifs')
dir_nwc = os.path.join(dir_base,'nwc')

os.makedirs(dir_skill,exist_ok=True)
os.makedirs(dir_nwc,exist_ok=True)
os.makedirs(dir_gif,exist_ok=True)

# Hard-coding some paths here to avoid potential pystepsrc issues.
data_src_radar = "rmi"
root_path = pysteps.rcparams.data_sources[data_src_radar]["root_path"]
path_fmt = pysteps.rcparams.data_sources[data_src_radar]["path_fmt"]
# BEWARE! This is not fixed in time. More recent radqpe files may have a different filename pattern.
fn_pattern = pysteps.rcparams.data_sources[data_src_radar]["fn_pattern"]
fn_ext = pysteps.rcparams.data_sources[data_src_radar]["fn_ext"]
importer_name = pysteps.rcparams.data_sources[data_src_radar]["importer"]
importer_kwargs = pysteps.rcparams.data_sources[data_src_radar]["importer_kwargs"]
timestep = pysteps.rcparams.data_sources[data_src_radar]["timestep"]

print("Started nowcast with:")
print(r' Startdate: %s' % startdate.strftime("%Y-%m-%d %H:%M"))
print(r' Forecast length: %i timesteps' % fc_length)
print(r' Number of ensemble members: %i' % nens)
print(r' Number of workers: %i' % ncores)
print(r' Rain/No-rain threshold: %.2f' % threshold)
print(r' Number of cascade levels: %i' % ncascade)
print(r' Motion vectors are loaded from: %s' % dir_motion)
print(r' Cascade decompositions are loaded from: %s' % dir_cascade)
print(r' NWP skill is saved in: %s' % dir_skill)
print(r' Nowcast netCDF file is saved in: %s' % dir_nwc)
print('')


# Load and preprocess the radar data

print('Loading and preprocessing radar analysis...')
fn_radar = pysteps.io.find_by_date(
        date = startdate,
        root_path = root_path,
        path_fmt = path_fmt,
        fn_pattern = fn_pattern,
        fn_ext = fn_ext,
        timestep = timestep,
        num_prev_files = 2
)

# Reading the radar hdf5 files with the appropriate importer
importer_radar = pysteps.io.get_method(importer_name,"importer")
r_radar, _, metadata_radar = pysteps.io.read_timeseries(
        inputfns = fn_radar,
        importer = importer_radar,
        legacy=False,
        **importer_kwargs
)

metadata_nwc = metadata_radar.copy()
metadata_nwc['shape'] = r_radar.shape[1:]

# 4. Prepare the radar analyses
converter = pysteps.utils.get_method("mm/h")
r_radar, metadata_radar = converter(r_radar,metadata_radar)

r_radar[r_radar < threshold] = 0.0
metadata_radar["threshold"] = threshold

r_obs = r_radar[-1,:,:].copy()
metadata_obs = metadata_radar.copy()

transformer = pysteps.utils.get_method("dB")
r_radar, metadata_radar = transformer(
        R = r_radar,
        metadata = metadata_radar,
        threshold = threshold,
#        zerovalue=-10.0
)

# Determine optical flow field with Lukas-Kanade
oflow_method = pysteps.motion.get_method("LK")
v_radar = oflow_method(r_radar)
print('done!')


# Get the available NWP dates, select the closest one and load the velocities and cascade


fcsttimes_nwp = []
for file in os.listdir(dir_motion):
    fcsttimes_nwp.append(
            datetime.datetime.strptime(file.split("_")[2].split('.')[0],'%Y%m%d%H%M%S')
    )

startdate_nwp = startdate + datetime.timedelta(minutes=timestep)
date_nwp = startdate_nwp + max([nwptime - startdate_nwp for nwptime in fcsttimes_nwp if nwptime <= startdate_nwp]) 

def load_NWP(model, date_nwp, dir_motion, dir_cascade):
    """
    Load the NWP cascade and velocities for the given model and date
    """
    fn_motion = os.path.join(dir_motion,
            r'motion_%s_%s.npy' % (model,date_nwp.strftime('%Y%m%d%H%M%S'))
    )
    fn_cascade = os.path.join(dir_cascade,
            r'cascade_%s_%s.nc' % (model,date_nwp.strftime('%Y%m%d%H%M%S'))
    )

    if not os.path.exists(fn_cascade):
        raise Exception('Cascade file %s accompanying motion file %s does not exist' % (fn_cascade,fn_motion))

    print(r'Loading NWP cascade and velocities for run started at %s...' % date_nwp.strftime('%Y-%m-%d %H:%M'))
    r_decomposed_nwp, v_nwp = pysteps.blending.utils.load_NWP(
            input_nc_path_decomp = fn_cascade,
            input_path_velocities = fn_motion,
            start_time=np.datetime64(startdate_nwp),
            n_timesteps=fc_length
    )

    return r_decomposed_nwp, v_nwp

models = ['ao13','ar13']
if len(models) == 1:
  r_decomposed_nwp, v_nwp = load_NWP(models[0], date_nwp, dir_motion, dir_cascade)
  r_decomposed_nwp = np.stack([r_decomposed_nwp], axis=0)
  v_nwp = np.stack([v_nwp],axis=0)
else:
  r_decomposed_nwp, v_nwp = zip(*(load_NWP(model, date_nwp, dir_motion, dir_cascade) for model in models))
  r_decomposed_nwp = np.stack(r_decomposed_nwp, axis=0)
  v_nwp = np.stack(v_nwp, axis=0)

print('done!')


# Prepare the netCDF exporter-function

def write_netCDF(R):
    R, _ = converter(R, metadata_radar)
    pysteps.io.export_forecast_dataset(R, exporter)

exporter = pysteps.io.initialize_forecast_exporter_netcdf(
        outpath = dir_nwc,
        outfnprefix = 'blended_nowcast_%s' % startdate.strftime("%Y%m%d%H%M"),
        startdate = startdate,
        timestep = timestep,
        n_timesteps = fc_length,
        shape = metadata_nwc['shape'],
        n_ens_members = nens,
        metadata = metadata_nwc,
        incremental = 'timestep'
)

# Start the nowcast

nwc_method = pysteps.blending.get_method("steps")
r_nwc = nwc_method(
        precip = r_radar,
        precip_models = r_decomposed_nwp,
        velocity = v_radar,
        velocity_models = v_nwp,
        timesteps = fc_length,
        timestep = timestep,
        issuetime = startdate,
        n_ens_members = nens,
        n_cascade_levels = ncascade,
        blend_nwp_members = False,
        precip_thr = metadata_radar['threshold'],
        kmperpixel = metadata_radar['xpixelsize']/1000.0,
        extrap_method = 'semilagrangian',
        decomp_method = 'fft',
        bandpass_filter_method = 'gaussian',
        noise_method = 'nonparametric',
        noise_stddev_adj = 'auto',
        ar_order = 2,
        vel_pert_method = None,
        weights_method = 'bps',
        conditional = False,
        probmatching_method = 'cdf',
        mask_method = 'incremental',
        smooth_radar_mask_range = 50,
        callback = write_netCDF,
        return_output = True,
        seed = 24,
        num_workers = ncores,
        fft_method = 'numpy',
        domain = 'spatial',
        outdir_path_skill = dir_skill,
        extrap_kwargs = None,
        filter_kwargs = None,
        noise_kwargs = None,
        vel_pert_kwargs = None,
        clim_kwargs = None,
        mask_kwargs = {"max_mask_rim" : 40 },
        measure_time = False
)


r_nwc, metadata_nwc = transformer(
        R = r_nwc,
        threshold = -10,
        inverse = True
)

pysteps.io.close_forecast_files(exporter)
print("nowcast done!")

