#!/usr/bin/python3
# conda activate r62GEDI
# python3 06_Processing_netCDF_files_into_np_arrays.py

# NOTES:
# This script reads the .netCDF files from the previous import step and 
# re-pacakges them as numpy .npy binary files in order to be used later 
# more easily in our analysis. 

import sys
sys.path.append('../')
from resources import read_pickle, int_to_month_name

import pandas as pd
import numpy as np
import xarray as xr
import os
import time
import warnings

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

##################################################################################################################

if rank == 0:
	MONTH = 'Feb'
elif rank == 1:
	MONTH = 'March'
elif rank == 2:
	MONTH = 'April'
elif rank == 3:
	MONTH = 'May'
elif rank == 4:
	MONTH = 'June'

OVERWRITE = False

directories = read_pickle('../.directories.pkl')

number_of_locations = 23
years = [2020, 2021, 2022, 2023]
output_path = os.path.join(directories['forecasts_path'], MONTH.capitalize())
summary_table_path = os.path.join(directories['path_to_breakup_data'], 'summary_table_locations.pkl')
summary_table_locations = pd.read_pickle(summary_table_path)
var_names = ['ssrd', 'mx2t24', 'mn2t24', 'sd', 'sf', 'rsn', 'tcwv', 'tp', 'msl', 'd2m'] # must be in correct order with 24H first and 6H last
path_to_daymet = os.path.join(directories['final_datasets_path'], f'DAYMET_{number_of_locations}_LOCATIONS_PRE_LSTM.pkl')
hist_daymet = pd.read_pickle(path_to_daymet)

##################################################################################################################

# creating final destination:
path_to_ensembles = os.path.join(directories['forecasts_path'], MONTH.capitalize(), 'processed/ensembles')
os.makedirs(path_to_ensembles, exist_ok=True)

# This section converts the ECMWF Seasonal Projections into numpy arrays with all ensembles
start_month_number = int_to_month_name([MONTH.capitalize()])[0]

start_time = time.time()
warnings.filterwarnings('ignore')
os.chdir(output_path)

for location in summary_table_locations.index:
    location = location.replace(' ', '_')
    location_files = [i for i in os.listdir() if i.endswith(location + '.nc')]

    six_H_data = xr.open_dataset([i for i in location_files if i.startswith('six_hour_interval')][0])
    twenty_four_H_data = xr.open_dataset([i for i in location_files if i.startswith('twenty_four_hour_interval')][0])

    all_6H = []
    all_24H = []

    for VAR in var_names:

        if VAR in ['msl', 'd2m']:

            arr = np.squeeze(six_H_data[VAR].to_numpy())
            all_6H.append(arr)

        else:

            arr = np.squeeze(twenty_four_H_data[VAR].to_numpy())
            all_24H.append(arr)

    all_6H = np.stack(all_6H)
    all_6H = np.moveaxis(all_6H, 0, -1)

    all_24H = np.stack(all_24H)
    all_24H = np.moveaxis(all_24H, 0, -1)

    # Averaging the 6 hour vars (sea level pressure and 2m dew point temp) then combining them
    all_6H = np.stack(np.split(all_6H, indices_or_sections = 4, axis = 2))
    all_6H = np.mean(all_6H, axis = 0)

    full = np.concatenate([all_24H, all_6H], axis = 3)

    # print('ECMWF MEDIUM RANGE FORECASTED ENSEMBLES SHAPE:', full.shape, 'FOR LOCATION:', location.replace('_', ' '))

    # THE VARIABLES ORDER FOR LAST DIMENSION OF full
    # 'surface_short_wave_radiation_downwards', 'tmax(deg c)', 'tmin(deg c)', 'snow_depth', 
    # 'snowfall', 'snow_density', 'column_water_vapour', 'prcp(mm/day)',
    # 'mean_sea_level_pressure', '2m_dew_point_temp'

    # converting temperatures to celsius
    full[:, :, :, 1] = full[:, :, :, 1] - 273.15
    full[:, :, :, 2] = full[:, :, :, 2] - 273.15
    full[:, :, :, 9] = full[:, :, :, 9] - 273.15

    # total precip in ECMWF is in m
    full[:, :, :, 7] = full[:, :, :, 7]*1000

    # solving cumulative issue:
    full[:, :, 1:, 7] = np.diff(full[:, :, :, 7], axis = 2)

    # as per ECMWF docs srrd should be divided by the accumulation period in seconds
    full[:, :, :, 0] = full[:, :, :, 0]/(3600*24)

    # solving cumulative issue:
    full[:, :, 1:, 0] = np.diff(full[:, :, :, 0], axis = 2)

    # to get swe (might need to check over this one, but units work)
    # I AM OVERWRITING THE SNOWFALL COLUMN HERE!!!!!!!
    full[:, :, :, 4] = full[:, :, :, 3]*full[:, :, :, 5] # 'snowfall' times 'snow_density'

    # creating vapor pressure from tmin:
    # I AM OVERWRITING THE 2m DEWPOINT COLUMN HERE!!!!!!!
    full[:, :, :, 9] = 1000*(6.1078*np.exp(np.divide((21.875*full[:, :, :, 2]), (full[:, :, :, 2] + 265.5))))
    # solving cumulative issue:
    full[:, :, 1:, 9] = np.diff(full[:, :, :, 9], axis = 2)

    # UPDATED VARIABLES ORDER FOR LAST DIMENSION OF full
    # 'surface_short_wave_radiation_downwards', 'tmax(deg c)', 'tmin(deg c)', 'snow_depth', 
    # 'swe', 'snow_density', 'column_water_vapour', 'prcp(mm/day)',
    # 'mean_sea_level_pressure', 'vp'

    npy_path = os.path.join(output_path, 'processed/', f'{location}.npy')

    if OVERWRITE:
        np.save(npy_path, full)
    elif (OVERWRITE == False) and (os.path.exists(npy_path)):
        pass
    elif (OVERWRITE == False) and (os.path.exists(npy_path) == False):
        np.save(npy_path, full)
	
print('#'*45)
print(f'CREATING NUMPY ARRAYS FOR EACH LOCATION COMPLETE FOR RANK {rank}', flush = True)
print('#'*45)
