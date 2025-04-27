#!/usr/bin/python3
# conda activate r62GEDI
# mpiexec -n 5 python3 07_Embed_forecast_data.py

# This script embeds the forecast data within the daymet data
# creating dfs that can be applied to the LSTM later on for 
# an additional evaluation step 

import sys
sys.path.append('../')
from resources import int_to_month_name, read_pickle

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
import warnings

directories = read_pickle('../.directories.pkl')

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
	
number_of_locations = 23
years = [2020, 2021, 2022, 2023]
output_path = os.path.join(directories['forecasts_path'], MONTH.capitalize())
summary_table_path = os.path.join(directories['path_to_breakup_data'], 'summary_table_locations.pkl')
summary_table_locations = pd.read_pickle(summary_table_path)
path_to_daymet = os.path.join(directories['final_datasets_path'], f'DAYMET_{number_of_locations}_LOCATIONS_PRE_LSTM.pkl')
hist_daymet = pd.read_pickle(path_to_daymet)

##################################################################################################################

# creating final destination:
path_to_ensembles = os.path.join(directories['forecasts_path'], MONTH.capitalize(), 'processed/ensembles')
os.makedirs(path_to_ensembles, exist_ok=True)

# This section converts the ECMWF Seasonal Projections into numpy arrays with all ensembles

start_month_number = int_to_month_name([MONTH.capitalize()])[0]

if rank == 0:
	start_time = time.time()

warnings.filterwarnings('ignore')

# creating the mins and maxes pkl which will have the shape (51, 10) 51: ensembles, 10: variables
# mins_maxes_pkl = {}
# mins_maxes_pkl['mins'] = np.min(np.stack(mins), axis = 0)
# mins_maxes_pkl['maxes'] = np.max(np.stack(maxes), axis = 0)
# write_pickle(f'{output_path}MINS_MAXES.pkl', mins_maxes_pkl)

# This section iterates over those numpy arrays and creates pickle files for each ensemble:
working_dir_path = os.path.join(output_path, 'processed/')
os.chdir(working_dir_path)

# getting a test array for viewing the shape:
test_arr = np.load([i for i in os.listdir() if i.endswith('.npy')][0])

# should be (51, 4, ~30*#months out, 10)
# 51 ensembles
# 4 years that we are observing (2020, 2021, 2022, 2023)
# 30*#months out so feb is 5 months to the end of the breakup season so 5*30 = 150
# 10 variables

test_shape = test_arr.shape
number_of_ensembles = test_shape[0]
forecast_window = test_shape[2]

# make sure to have correct order of locations:
# creating the complete daymet df:

# you must use the Daymet mins and maxes for normalization:
path_to_daymet = os.path.join(directories['final_datasets_path'], f'DAYMET_{number_of_locations}_LOCATIONS_PRE_LSTM.pkl')
mins_maxes_df = pd.read_pickle(path_to_daymet)['mins_maxes_df']
mins = mins_maxes_df['mins']
maxes = mins_maxes_df['maxes']

# We must re-arrange the above variables from Daymet to match the new order of terms of the ECMWF variables above
# there are variables in the above full array of ECMWF that we will not use and an equal number of useless vars
# in the daymet min max info. Therefore the useful vars will be set to overlap one another and the useless ones
# will also do the same.
mins = np.array(mins.loc[['srad(W/m^2)', 'tmax(deg c)', 'tmin(deg c)', 'Year', 'swe(kg/m^2)', 'dayl(s)', 'COS_Radians', 'prcp(mm/day)', 'SIN_Radians', 'vp(Pa)', ]])
maxes = np.array(maxes.loc[['srad(W/m^2)', 'tmax(deg c)', 'tmin(deg c)', 'Year', 'swe(kg/m^2)', 'dayl(s)', 'COS_Radians', 'prcp(mm/day)', 'SIN_Radians', 'vp(Pa)', ]])

if rank == 0:
	print('PROCESSING YEARS:', years)

years_with_year_before = [years[0] - 1] + years

for ens in range(number_of_ensembles):
	df = hist_daymet['df']
	df['Location'] = np.concatenate([hist_daymet['train_location'], hist_daymet['val_location'], hist_daymet['test_location']])
	df = df[df.index.year.isin(years_with_year_before)]
	
	# these are the holdouts so they won't be in the main Daymet DF
	holdout_locations = list(summary_table_locations.loc[summary_table_locations['Holdouts'] == True].index)
	holdout_locations = [i.replace(' ', '_') for i in holdout_locations]

	for location in summary_table_locations.index:
		
		location = location.replace(' ', '_')
		
		if location in holdout_locations:
			continue
		
		# importing and normalizing arrays using Daymet mins and maxes
		arr = np.load(location + '.npy')[ens]
		arr = (arr - mins)/(maxes - mins)
		for YEAR_IDX, YEAR in enumerate(years):
				
			mask = (df.Location == location) & (df.index > f'0{start_month_number}-01-{YEAR}') & (df.index <= (datetime.strptime(f'0{start_month_number}-01-{YEAR}', '%m-%d-%Y') +  timedelta(days=forecast_window)))
			
			df.loc[mask, 'prcp(mm/day)'] = arr[YEAR_IDX, :, 7]
			df.loc[mask, 'srad(W/m^2)'] = arr[YEAR_IDX, :, 0]
			df.loc[mask, 'swe(kg/m^2)'] = arr[YEAR_IDX, :, 4]
			df.loc[mask, 'tmax(deg c)'] = arr[YEAR_IDX, :, 1]
			df.loc[mask, 'tmin(deg c)'] = arr[YEAR_IDX, :, 2]
			df.loc[mask, 'vp(Pa)'] = arr[YEAR_IDX, :, 9]

	sub_df_path = os.path.join(directories['forecasts_path'], MONTH.capitalize(), 'processed/ensembles/', f'ENSEMBLE_{ens}.pkl')
	df.to_pickle(sub_df_path)
	print('ENSEMBLE:', ens, 'MONTH:', MONTH, 'COMPLETE', flush = True)

comm.barrier()

if rank == 0:
	stop_time = time.time()
	complete_time = stop_time - start_time	
	complete_time = round((complete_time/60), 2)
	print('PROCESS COMPLETE TOOK:', complete_time, 'MINUTES')



