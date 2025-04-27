#!/usr/bin/python3
# conda activate DL
# python3 02_holdouts_val_naive_predictions.py

# This script runs the model on the validation data in order to use
# it on a plot later on (not in the manuscript); 
# the holdouts dataset, for comparison to data 
# unseen by the model; and creates/runs a naive model,
# that uses the previous breakup date for a given location to determine the 
# subsequent breakup date. The results from the naive model will be 
# printed to the output at the end step so copy and paste the results when
# it is finished. 

import pandas as pd
import numpy as np
import sys
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#################################################
# Script Parameters:
lstm_model_path = '/mnt/locutus/remotesensing/r62/river_ice_breakup/Results/DAYMET_Results/Models/DAYMET/DAYMET_BAYESIAN'
final_datasets_path = '/mnt/locutus/remotesensing/r62/river_ice_breakup/final_Daymet_datasets'
results_path = '/mnt/locutus/remotesensing/r62/river_ice_breakup/Results/DAYMET_Results/Output'
LOOKBACK_WINDOW = 457
number_of_locations_in_main_df = 23
number_of_locations_in_holdouts = 10
OVERWRITE = True
#################################################
def load_model(path):
    return tf.keras.models.load_model(path)

sys.path.append('../')
from resources import df_to_LSTM

model = load_model(lstm_model_path)
# This portion I will run the model on the validation data (used in one
# of the figures)

data_pkl = pd.read_pickle(f'{final_datasets_path}/DAYMET_{number_of_locations_in_main_df}_LOCATIONS_PRE_LSTM.pkl')

val_df = data_pkl['val_df']
val_y = data_pkl['val_y']
val_dates = data_pkl['val_dates']

# plot_data = val_df.iloc[LOOKBACK_WINDOW:, :][['tmax(deg c)', 'tmin(deg c)']]

my_lambda = lambda x: x.replace('_', ' ')
remov_underscores = np.vectorize(my_lambda)
sites = remov_underscores(data_pkl['val_location'][LOOKBACK_WINDOW:])

if (os.path.exists(f'{results_path}/Main/val_predictions.npy')) and (OVERWRITE == False):
    predictions = np.load(f'{results_path}/Main/val_predictions.npy')
else:
    data = df_to_LSTM(val_df, LOOKBACK_WINDOW)
    predictions = model.predict(data)
    predictions = np.squeeze(predictions)
    np.save(f'{results_path}/Main/val_predictions.npy', predictions)

# plot_data['sites'] = sites
# plot_data['actuals'] = val_y[LOOKBACK_WINDOW:]
# plot_data['preds'] = predictions[1:]
# plot_data['dates'] = val_dates[LOOKBACK_WINDOW:]
# plot_data['dates'] = pd.to_datetime(plot_data['dates'])

# plot_data['Predicted_Breakup'] = 0
# plot_data.reset_index(inplace=True, drop=True)
# plot_data[['tmax(deg c)', 'tmin(deg c)']] = plot_data[['tmax(deg c)', 'tmin(deg c)']].astype('float32')

# plot_data.to_pickle(f'{results_path}/Main/validation_plot_data.pkl')

print('EVALUATING VALIDATION DATASET COMPLETE')

# This portion I will run the model on the holdout locations
if (os.path.exists(f'{results_path}/Holdouts/holdout_predictions_1980-2023.npy')) and (OVERWRITE == False):
	pass
else:
	data = pd.read_pickle(f'{final_datasets_path}/DAYMET_{number_of_locations_in_holdouts}_HOLDOUTS_PRE_LSTM.pkl')
	data = df_to_LSTM(data['df'], LOOKBACK_WINDOW)
	holdout_preds = np.squeeze(model.predict(data))
	np.save(f'{results_path}/Holdouts/holdout_predictions_1980-2023.npy', holdout_preds)

print('EVALUATING HOLDOUT DATASET COMPLETE')

# Naive model evaluation

daymet_data = pd.read_pickle('/mnt/locutus/remotesensing/r62/river_ice_breakup/final_Daymet_datasets/DAYMET_20_LOCATIONS_PRE_LSTM.pkl')
print(daymet_data.keys())

val_df = daymet_data['val_df']
val_df['locations'] = daymet_data['val_location']

test_df = daymet_data['test_df']
test_df['locations'] = daymet_data['test_location']
test_years = test_df.index.year.unique()

val_test_df = pd.concat([val_df, test_df], axis = 0)

val_test_df['actuals'] = np.concatenate([daymet_data['val_y'], daymet_data['test_y']])
val_test_df.reset_index(inplace = True)
val_test_df.rename(columns = {'index':'dates'}, inplace = True)

unique_locations = val_test_df.locations.unique()
val_test_df = val_test_df[val_test_df.actuals == 1]

diffs = []
mapes = []

for location in unique_locations:
    temp_df = val_test_df[val_test_df.locations == location]['dates']

    for idx in range(1, len(temp_df)):
        current_date = temp_df.iloc[idx]
        previous_date = temp_df.iloc[idx-1]

        if previous_date.year == current_date.year - 1:
            diff = current_date.dayofyear - previous_date.dayofyear
            diffs.append(diff)

            mape = np.abs(diff/current_date.dayofyear)
            mapes.append(mape)

        else:
            diffs.append(np.nan)

print('MAE NAIVE MODEL:', round(np.nanmean(np.abs(np.array(diffs))), 3))
print('SDAR NAIVE MODEL:', round(np.nanstd(np.abs(np.array(diffs))), 3))
print('MAX NAIVE MODEL:', round(np.nanmax(np.abs(np.array(diffs))), 3))
print('MIN NAIVE MODEL:', round(np.nanmin(np.abs(np.array(diffs))), 3))
print('MAPE NAIVE MODEL:', round(100*np.nanmean(np.abs(np.array(mapes))), 3))



















