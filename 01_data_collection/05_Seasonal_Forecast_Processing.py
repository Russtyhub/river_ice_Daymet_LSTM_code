#!/usr/bin/python3
# conda activate r62GEDI
# mpiexec -n 5 python3 Seasonal_Forecast_Processing.py

# This script is intended to process the data imported from Seasonal_Forecast_API_Request.py

import sys
sys.path.append('/home/r62/repos/russ_repos/Functions')
from STANDARD_FUNCTIONS import read_pickle
from TIME import int_to_month_name

import pandas as pd
import numpy as np
from STANDARD_FUNCTIONS import runcmd, write_pickle
import xarray as xr
import os
import time
from datetime import datetime, timedelta
import time
import warnings
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

##################################################################################################################

if rank == 0:
	MONTH = 'feb'
elif rank == 1:
	MONTH = 'march'
elif rank == 2:
	MONTH = 'april'
elif rank == 3:
	MONTH = 'may'
elif rank == 4:
	MONTH = 'june'
	
years = [2020, 2021, 2022, 2023]
output_path = f'/mnt/locutus/remotesensing/r62/river_ice_breakup/ERA5/monthly_forecasts/differing_spans/{MONTH.lower()}/' # Must have last slash
locations_csv = pd.read_csv('/mnt/locutus/remotesensing/r62/river_ice_breakup/Breakup_Data/Daymet_analysis/02_25_Sites_Info.csv')
var_names = ['ssrd', 'mx2t24', 'mn2t24', 'sd', 'sf', 'rsn', 'tcwv', 'tp', 'msl', 'd2m'] # must be in correct order with 24H first and 6H last
hist_daymet = read_pickle('/mnt/locutus/remotesensing/r62/river_ice_breakup/final_Daymet_datasets/DAYMET_20_LOCATIONS_PRE_LSTM.pkl')

##################################################################################################################

# creating final destination:
if not os.path.exists(f'/mnt/locutus/remotesensing/r62/river_ice_breakup/ERA5/monthly_forecasts/differing_spans/{MONTH.lower()}/processed/ensembles'):
	os.makedirs(f'/mnt/locutus/remotesensing/r62/river_ice_breakup/ERA5/monthly_forecasts/differing_spans/{MONTH.lower()}/processed/ensembles')

# This section converts the ECMWF Seasonal Projections into numpy arrays with all ensembles

start_month_number = int_to_month_name([MONTH.capitalize()])[0]

start_time = time.time()
warnings.filterwarnings('ignore')

os.chdir(output_path)

# This is probably unnecessary
# mins = []
# maxes = []

for location in locations_csv['Site']:
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
		
	# mins.append(np.min(full, axis=(1, 2)))
	# maxes.append(np.max(full, axis=(1,2)))

	# UPDATED VARIABLES ORDER FOR LAST DIMENSION OF full
	# 'surface_short_wave_radiation_downwards', 'tmax(deg c)', 'tmin(deg c)', 'snow_depth', 
	# 'swe', 'snow_density', 'column_water_vapour', 'prcp(mm/day)',
	# 'mean_sea_level_pressure', 'vp'

	np.save(f'{output_path}/processed/{location}.npy', full)
	
print('#'*45)
print(f'CREATING NUMPY ARRAYS FOR EACH LOCATION COMPLETE FOR RANK {rank}', flush = True)
print('#'*45)

# creating the mins and maxes pkl which will have the shape (51, 10) 51: ensembles, 10: variables
# mins_maxes_pkl = {}
# mins_maxes_pkl['mins'] = np.min(np.stack(mins), axis = 0)
# mins_maxes_pkl['maxes'] = np.max(np.stack(maxes), axis = 0)
# write_pickle(f'{output_path}MINS_MAXES.pkl', mins_maxes_pkl)

# This section iterates over those numpy arrays and creates pickle files for each ensemble:
os.chdir(output_path + 'processed/')

# getting a test array for viewing the shape:
test_arr = np.load([i for i in os.listdir() if i.endswith('.npy')][0])

# should be (51, 4, 30*#months out, 10)
# 51 ensembles
# 4 years that we are observing (2020, 2021, 2022, 2023)
# 30*#months out so feb is 5 months to the end of the breakup season so 5*30 = 150
# 10 variables

test_shape = test_arr.shape
number_of_ensembles = test_shape[0]

# make sure to have correct order of locations:
# creating the complete daymet df:

# you must use the Daymet mins and maxes for normalization:
mins = read_pickle('/mnt/locutus/remotesensing/r62/river_ice_breakup/final_Daymet_datasets/DAYMET_20_LOCATIONS_MINS_MAXES.pkl')['mins']
maxes = read_pickle('/mnt/locutus/remotesensing/r62/river_ice_breakup/final_Daymet_datasets/DAYMET_20_LOCATIONS_MINS_MAXES.pkl')['maxes']

# We must re-arrange the above variables from Daymet to match the new order of terms of the ECMWF variables above
# there are variables in the above full array of ECMWF that we will not use and an equal number of useless vars
# in the daymet min max info. Therefore the useful vars will be set to overlap one another and the useless ones
# will also do the same.
mins = np.array(mins.loc[['srad(W/m^2)', 'tmax(deg c)', 'tmin(deg c)', 'Year', 'swe(kg/m^2)', 'dayl(s)', 'COS_Radians', 'prcp(mm/day)', 'SIN_Radians', 'vp(Pa)', ]])
maxes = np.array(maxes.loc[['srad(W/m^2)', 'tmax(deg c)', 'tmin(deg c)', 'Year', 'swe(kg/m^2)', 'dayl(s)', 'COS_Radians', 'prcp(mm/day)', 'SIN_Radians', 'vp(Pa)', ]])

print('PROCESSING YEARS:', years)
years_with_year_before = [years[0] - 1] + years

for ens in range(number_of_ensembles):
	df = hist_daymet['df']
	df['Location'] = np.concatenate([hist_daymet['train_location'], hist_daymet['val_location'], hist_daymet['test_location']])
	df = df[df.index.year.isin(years_with_year_before)]
	
	for location in locations_csv['Site']:
		location = location.replace(' ', '_')
		
		# these are the holdouts so they won't be in the main Daymet DF
		if location in ['Ambler_Kobuk_River', 'Bettles_Koyukuk_River', 'Eagle_Yukon_River', 'Kaltag_Yukon_River', 'Nikolai_Kuskokwim_River']:
			continue
		
		arr = np.load(location + '.npy')[ens]
		# print(arr.shape)
		# mins = mins_maxes_pkl['mins'][ens]
		# maxes = mins_maxes_pkl['maxes'][ens]
		
		# normalize array
		arr = (arr - mins)/(maxes - mins)
		
		# print('MIN:', np.nanmin(arr[:, :, 1]), 'MAX:', np.nanmax(arr[:, :, 1]))
		
		# split the array by the four years
		# arr_split_by_year = np.split(arr, indices_or_sections = 4, axis = 0)
		
		for YEAR_IDX, YEAR in enumerate(years):
				
			q_mask = df['prcp(mm/day)'][(df.Location == location) &
							   (df.index > f'0{start_month_number}-01-{YEAR}') &
							   (df.index <= (datetime.strptime(f'0{start_month_number}-01-{YEAR}', '%m-%d-%Y') +  timedelta(days=test_shape[2])))]
						  
			# print(q_mask.shape, arr[YEAR_IDX, :, 7].shape)
				
			df.loc[(df.Location == location) &
							   (df.index > f'0{start_month_number}-01-{YEAR}') &
							   (df.index <= (datetime.strptime(f'0{start_month_number}-01-{YEAR}', '%m-%d-%Y') +  timedelta(days=test_shape[2]))), 'prcp(mm/day)'] = arr[YEAR_IDX, :, 7]
				
			df.loc[(df.Location == location) &
							   (df.index > f'0{start_month_number}-01-{YEAR}') &
							   (df.index <= (datetime.strptime(f'0{start_month_number}-01-{YEAR}', '%m-%d-%Y') +  timedelta(days=test_shape[2]))), 'srad(W/m^2)'] = arr[YEAR_IDX, :, 0]

			df.loc[(df.Location == location) &
							   (df.index > f'0{start_month_number}-01-{YEAR}') &
							   (df.index <= (datetime.strptime(f'0{start_month_number}-01-{YEAR}', '%m-%d-%Y') +  timedelta(days=test_shape[2]))), 'swe(kg/m^2)'] = arr[YEAR_IDX, :, 4]
			
			df.loc[(df.Location == location) &
							   (df.index > f'0{start_month_number}-01-{YEAR}') &
							   (df.index <= (datetime.strptime(f'0{start_month_number}-01-{YEAR}', '%m-%d-%Y') +  timedelta(days=test_shape[2]))), 'tmax(deg c)'] = arr[YEAR_IDX, :, 1]
			
			df.loc[(df.Location == location) &
							   (df.index > f'0{start_month_number}-01-{YEAR}') &
							   (df.index <= (datetime.strptime(f'0{start_month_number}-01-{YEAR}', '%m-%d-%Y') +  timedelta(days=test_shape[2]))), 'tmin(deg c)'] = arr[YEAR_IDX, :, 2]
			
			df.loc[(df.Location == location) &
							   (df.index > f'0{start_month_number}-01-{YEAR}') &
							   (df.index <= (datetime.strptime(f'0{start_month_number}-01-{YEAR}', '%m-%d-%Y') +  timedelta(days=test_shape[2]))), 'vp(Pa)'] = arr[YEAR_IDX, :, 9]
			

	df.to_pickle(f'/mnt/locutus/remotesensing/r62/river_ice_breakup/ERA5/monthly_forecasts/differing_spans/{MONTH.lower()}/processed/ensembles/ENSEMBLE_{ens}.pkl')
	print('ENSEMBLE:', ens, 'MONTH:', MONTH, 'COMPLETE', flush = True)
	
stop_time = time.time()
complete_time = stop_time - start_time	
complete_time = round((complete_time/60), 2)
print('PROCESS COMPLETE TOOK:', complete_time, 'MINUTES')



