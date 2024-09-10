#!/usr/bin/python3
# conda activate DL
# mpiexec -n 5 python3 Forecast_Ensembles.py

# This script takes each processed dataset by ensemble and forecast span, creates a likelihood function 
# for it, then predicts breakup on it by embedding that portion of the likelihood function into the larger 
# daymet dataset. Also, this script applies a separately trained model for each year given the script:
# 01_iterative_LSTM_evaluation.py This means that the best model after training, right up until the year
# being tested, is implimented. The script runs in parallel for the different forecast ranges by the 
# month that the forecast range starts at.

import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import sys
import copy 
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sys.path.append('../../Functions')
from TF_FUNCTIONS import df_to_LSTM
from STANDARD_FUNCTIONS import write_pickle
from TIME import int_to_month_name

########################################################################################################

# Before I was using the daymet model as it was from the end of training (end of validation year:2013)
# daymet_model = load_model('/mnt/locutus/remotesensing/r62/river_ice_breakup/Results/DAYMET_Results/Models/DAYMET/DAYMET_BAYESIAN')
iterative_models_path = '/mnt/locutus/remotesensing/r62/river_ice_breakup/Results/DAYMET_Results/Models/temp_iterative_evaluation/iterative_models'
daymet_data = pd.read_pickle('/mnt/locutus/remotesensing/r62/river_ice_breakup/final_inputs/DAYMET_20_LOCATIONS_PRE_LSTM.pkl')
daymet_df = daymet_data['df']
years_in_ensemble_dfs = [2020, 2021, 2022, 2023]
CREATE_MEAN_ENSEMBLE = True
NUMBER_OF_ENSEMBLES = 51
LOOKBACK_WINDOW = 457
########################################################################################################

base_directory = '/mnt/locutus/remotesensing/r62/river_ice_breakup/ERA5/monthly_forecasts/differing_spans' # no slash

def get_breakup_dates_by_yr(data_by_site):
    d = copy.copy(data_by_site)
    year = d.set_index('dates').index.year
    d['year'] = year
    predicted_breakup_dates = []
    for yr in np.sort(np.unique(np.array(year))):
        data_for_year = d[d['year'] == yr]
        predicted_breakup_dates.append(data_for_year[data_for_year['preds'] == data_for_year['preds'].max()]['dates'].dt.date)
    return np.squeeze(np.array(predicted_breakup_dates))

def date_difference(list1, list2, absolute):
    result = []
    for date1 in list1:
        year1 = date1.year
        for date2 in list2:
            year2 = date2.year
            if year1 == year2:
                
                if absolute:
                    difference = abs(date1 - date2).days
                else:
                    difference = (date1 - date2).days
                result.append(difference)
                break
    return np.array(result)

def extract_values(years, dates, values):
    result = []

    for year in years:
        matching_dates = [val for val, dt in zip(values, dates) if dt.year == year]

        if matching_dates:
            result.append(matching_dates[0])
        else:
            result.append(np.nan)

    return result

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

month_num = int_to_month_name([MONTH.capitalize()])[0]
	
dfs = []

if CREATE_MEAN_ENSEMBLE:
	for ENSEMBLE in range(NUMBER_OF_ENSEMBLES):

		if ENSEMBLE == 0:
			col_names = pd.read_pickle(f'{base_directory}/{MONTH}/processed/ensembles/ENSEMBLE_{ENSEMBLE}.pkl').columns

		ens_x = np.array(pd.read_pickle(f'{base_directory}/{MONTH}/processed/ensembles/ENSEMBLE_{ENSEMBLE}.pkl'))

		if ENSEMBLE == 0:
			locations = np.squeeze(ens_x[:, -1])

		ens_x = ens_x[:, 0:-1]
		dfs.append(ens_x)

	dfs = np.stack(dfs)
	mean_model = np.mean(dfs, axis = 0)
	del dfs

	mean_model = pd.DataFrame(mean_model)
	mean_model['Locations'] = locations
	mean_model.columns = col_names
	
	# convert all columns to float32 except Locations
	mean_model[col_names[0:-1]] = mean_model[col_names[0:-1]].astype('float32')
	# save averaged model as ensemble mean
	mean_model.to_pickle(f'{base_directory}/{MONTH}/processed/ensembles/ENSEMBLE_MEAN.pkl')

	print('MEAN ENSEMBLE CREATED FOR MONTH:', MONTH, 'TO JULY', flush = True)
	
locations_list = []

ENSEMBLES = [i for i in range(NUMBER_OF_ENSEMBLES)] + ['MEAN']

for ENSEMBLE in ENSEMBLES:
	
	ensemble_df_list = [] # pd.DataFrame(columns = ['Location', '2018', '2019', '2020', '2021', '2022'] )
	
	# collecting ensemble data and finding forecasted liklihood function:
	ens_x = pd.read_pickle(f'{base_directory}/{MONTH}/processed/ensembles/ENSEMBLE_{ENSEMBLE}.pkl').drop('Location', axis = 1)
	ens_x_lstm = df_to_LSTM(ens_x, LOOKBACK_WINDOW)
	ens_x_lstm = tf.convert_to_tensor(ens_x_lstm, dtype='float32')
      
	# I remove the lookback window to align ens_x with the ens_x_lstm object
	ens_x_shortended = ens_x.iloc[LOOKBACK_WINDOW-1:, :]
          
	forecast_predictions = []
	for year in years_in_ensemble_dfs:
            
		# need to create a mask to only operate on the year being iterated on
		year_mask = ens_x_shortended.index.year == year
		
		daymet_model = tf.keras.models.load_model(f'{iterative_models_path}/best_model_{year}.keras')
		forecast_preds = np.squeeze(daymet_model.predict(ens_x_lstm[year_mask], verbose = 0))
		forecast_predictions.append(forecast_preds)

	forecast_predictions = np.concatenate(forecast_predictions)
	del ens_x_lstm, ens_x_shortended
	
	# Adding vars back: sites, actuals, dates
	ens_x['sites'] = np.concatenate([daymet_data['train_location'], daymet_data['val_location'], daymet_data['test_location']])[daymet_df.index.year.isin(years_in_ensemble_dfs)]
	ens_x['actuals'] = np.concatenate([daymet_data['train_y'], daymet_data['val_y'], daymet_data['test_y']])[daymet_df.index.year.isin(years_in_ensemble_dfs)]
	ens_x['dates'] = np.concatenate([daymet_data['train_dates'], daymet_data['val_dates'], daymet_data['test_dates']])[daymet_df.index.year.isin(years_in_ensemble_dfs)]
	
	# adding in the forecast liklihood function
	ens_x = ens_x.iloc[LOOKBACK_WINDOW:, :]
	ens_x['preds'] = forecast_predictions
	
	# reindexng and removing excess vars:
	ens_x.reset_index(inplace = True, drop = True)
	ens_x = ens_x[['sites', 'actuals', 'preds', 'dates']]
	ens_x.sort_values(by = ['sites', 'dates'], inplace = True)
	
	# adding predicted date placeholder:
	ens_x['Predicted_Breakup'] = 0
		
	for DF in ens_x.groupby('sites'):
		
		location = DF[0]
		data_by_site = DF[1]
		data_by_site = data_by_site.sort_values('dates')

		# Getting true breakup dates and expected:
		true_break_up_dates = np.array(data_by_site['dates'][data_by_site['actuals'] == 1].dt.date)
		predicted_break_up_dates = get_breakup_dates_by_yr(data_by_site)

		# Find the differences in true vs predicted:
		differences = np.array(date_difference(predicted_break_up_dates, true_break_up_dates, False))
		
		# if the breakup happened before this monthly time frame (ie MONTH - July 1st) then that observation must be removed
		mask = np.array([j.month >= month_num for j in true_break_up_dates])
		
		# creating a key for each location within a dictionary since each location has a different number of breakup events
		diffs_to_keep = differences[mask]
		breakup_dates_to_keep = true_break_up_dates[mask]
		
		new_row = [location] + extract_values(years_in_ensemble_dfs, breakup_dates_to_keep, diffs_to_keep)
		ensemble_df_list.append(new_row)
		
	ensemble_df_list = pd.DataFrame(np.stack(ensemble_df_list))
	ensemble_df_list.columns = ['Location'] + years_in_ensemble_dfs
	
	ensemble_df_list.to_pickle(f'{base_directory}/results/{MONTH}/ENSEMBLE_{ENSEMBLE}.pkl')
	print('ENSEMBLE', ENSEMBLE, 'STARTING ON MONTH', MONTH, 'TO JULY COMPLETED', flush = True)
	
