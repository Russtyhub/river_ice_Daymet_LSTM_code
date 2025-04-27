#!/usr/bin/python3
# conda activate DL
# Calculates Shapley scores for the River Ice LSTM Model:
# mpiexec -n 15 python3 temporal_shaps.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import tensorflow as tf
import random
import warnings
import numpy as np
import copy
import pandas as pd
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sys.path.append('../')
from resources import make_lists_equal_length, df_to_LSTM

def load_model(path):
    return tf.keras.models.load_model(path)

AGGREGATE_LOCATIONS = False
DIAGNOSE = False
WINDOW_SIZE = 457
M = 500 # Monte Carlo simulation # of iterations
number_of_locations = 23
location_to_operate_on = 'Fort Yukon Yukon River'

input_data = pd.read_pickle(f'/mnt/locutus/remotesensing/r62/river_ice_breakup/final_Daymet_datasets/DAYMET_{number_of_locations}_LOCATIONS_PRE_LSTM.pkl')
output_path = '/mnt/locutus/remotesensing/r62/river_ice_breakup/Results/DAYMET_Results/Output/Shapley_Analysis/temp_store_columns'
models_path = '/mnt/locutus/remotesensing/r62/river_ice_breakup/Results/DAYMET_Results/Models/temp_iterative_evaluation/iterative_models'

# importing training data
train_df = input_data['train_df']
train_dates = input_data['train_dates']
train_location = input_data['train_location']

# importing testing data
test_df = input_data['test_df']
test_locations = input_data['test_location'] 
test_dates = input_data['test_dates'] 
# test_X = df_to_LSTM(test_df, window_size=WINDOW_SIZE)

# for naming the files later on:
alternative_col_names = ['year', 'daylight', 'precip', 'srad', 
                         'swe', 'tmax', 'tmin', 'vp', 'water_frac', 
                         'twi_mean', 'perimeter', 'gravelius', 
                         'cslope', 'cos_rads', 'sin_rads']

## We must aggregate over the locations
### First the test dataset
test_df['location'] = test_locations
column_names = list(test_df.columns)
column_names.pop(-1)

if AGGREGATE_LOCATIONS:
    locations_data = []
    for idx, subset in enumerate(test_df.groupby('location')):        
        DF = subset[1]
        DF.sort_index(inplace=True)
        DF.drop('location', axis=1, inplace=True)
        
        if idx == 0: # I noted that the first location is one of the shortest DFs
            # Else this would not work...
            index_to_match = DF.index
        
        DF = DF.loc[index_to_match]
        locations_data.append(np.array(DF))
        
    locations_data = np.stack(locations_data)
    locations_data = np.nanmean(locations_data, axis = 0) # averaging by location
    locations_data = pd.DataFrame(locations_data)
    locations_data.columns = column_names
    locations_data.index = index_to_match
    test_df_avg = locations_data
    del locations_data

else:
    location_to_operate_on = location_to_operate_on.replace(' ', '_')
    test_df_avg = test_df[test_df['location'] == location_to_operate_on]
    test_df_avg = test_df_avg.drop(['location'], axis=1).copy()
    index_to_match = test_df_avg.index

### Second the train dataset
train_df['location'] = train_location
column_names = list(train_df.columns)
column_names.pop(-1)

if AGGREGATE_LOCATIONS:
    for idx, subset in enumerate(train_df.groupby('location')):        
        DF = subset[1]
        DF.sort_index(inplace=True)
        if idx == number_of_locations - 1: # I noted that the last location is one of the shortest DFs
            # Else this would not work...
            index_to_match = DF.index
        
    locations_data = []
    for idx, subset in enumerate(train_df.groupby('location')):        
        DF = subset[1]
        DF.sort_index(inplace=True)
        DF.drop('location', axis=1, inplace=True)
        DF = DF.loc[index_to_match]
        DF[DF == -1] = np.nan # Here I have to remove the masked values
        locations_data.append(np.array(DF))
        
    locations_data = np.stack(locations_data)
    locations_data = np.nanmean(locations_data, axis = 0) # averaging by location
    locations_data = pd.DataFrame(locations_data)
    locations_data.columns = column_names
    locations_data.index = index_to_match
    train_df_avg = locations_data
    del locations_data

else:
    train_df_avg = train_df[train_df['location'] == location_to_operate_on]
    train_df_avg = train_df_avg.drop(['location'], axis=1).copy()

# I chose to trim the training dataset
# so it is more representative of the information impacting
# the test data:
train_df_avg = train_df_avg[train_df_avg.index.year >= 1997]

# convert data to LSTM ready format:
train_X = df_to_LSTM(train_df_avg, window_size=WINDOW_SIZE)
test_X = df_to_LSTM(test_df_avg, window_size=WINDOW_SIZE)

# setting the features to be used
n_features = train_df_avg.shape[1]
feature_idxs = list(range(n_features))

# modifying test and train dfs and finding indices for later
test_df_avg = test_df_avg.iloc[WINDOW_SIZE:, :]
train_df_avg = train_df_avg.iloc[WINDOW_SIZE:, :]

test_dates = test_df_avg.index
train_dates = train_df_avg.index

if (rank == 0) and DIAGNOSE:
    print('SHAPE OF TRAIN DF:', train_df_avg.shape, flush = True)
    print('SHAPE OF TEST DF:', test_df_avg.shape, flush = True)
    print('SHAPE OF test_X:', test_X.shape, flush = True)
    print('COLUMNS IN TRAIN DF:', train_df_avg.columns, flush = True) # turn on after diagnostics

# find the mins and maxes of each respective model (by year)
# to norm the probs
if rank == 0:
    print()
    print('CALCULATING THE MINS AND MAXES OF THE ITERATIVE LSTMs:', flush = True)
    print()
min_max_dict = {}
for year in test_df_avg.index.year.unique():
    model = load_model(f'{models_path}/best_model_{year}.keras')
    pred_probs_train = np.squeeze(model.predict(train_X, verbose = 0))
    min_max_dict[year] = (np.min(pred_probs_train), np.max(pred_probs_train))

# function to calculate probs at each iteration
# also normalizes probabilities since they 
# vary by year:
def return_prob(arr, year):
    mins_maxes = min_max_dict[year]
    probs = np.squeeze(model.predict(arr, verbose = 0))
    probs = (probs - mins_maxes[0]) / (mins_maxes[1] - mins_maxes[0])
    return probs

if rank == 0:
    start = time.time()
    print()
    print('COMPUTING SHAPLEY SCORES', flush = True)

shapley_scores = []
for j in range(n_features):
    
    if j != rank:
        continue

    column_name = alternative_col_names[j]
    print(f'RANK {rank} OPERATING ON COLUMN INDEX {j} {column_name}', flush = True)
    feature_idxs.remove(j)

    previous_year = None
    for sample_idx in range(test_df_avg.shape[0]):

        if (rank == 0) and (sample_idx%100 == 0):
            print(f'COMPLETED {sample_idx} ITERATIONS', flush=True)

        x_date = test_dates[sample_idx]
        x_year = x_date.year

        if previous_year is not None and x_year != previous_year:
            model = load_model(f'{models_path}/best_model_{x_year}.keras')
        previous_year = x_year

        x = test_X[sample_idx]
        candidates = np.where((train_dates.month == x_date.month) & (train_dates.day == x_date.day))[0] 
        zip_Ms_cand_range = make_lists_equal_length(list(range(M)), list(range(len(candidates))))
        
        x_plus_j_list = []
        x_minus_j_list = []

        for _, candidate in zip_Ms_cand_range:

            # temporaly based sample from the train data
            z = train_X[candidates[candidate]]

            # feature_idxs is list of 0 - 29. We take a random sample of those features where the number
            # we select is equal to the equation below. x_idx is a random selection of 6 features indices
            # Example: [20, 23, 0, 9, 21, 24]
            x_idx = random.sample(feature_idxs, min(max(int(0.2*n_features), random.choice(feature_idxs)), int(0.8*n_features)))

            # z_idx are the indices from feature_idxs not in x_idx. In our example:
            # [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 22, 25, 26, 27, 28, 29]
            z_idx = np.setdiff1d(np.array(feature_idxs), x_idx)

            # construct two new instances

            # x_plus_j:
            # Here we iterate through all of the original column indices
            # if the index is in x_idx (with the column we removed added back in)
            # we use the column of the test data else we use the column from the random training observation
            x_plus_j = np.array([x[:, col] if col in x_idx + [j] else z[:, col] for col in range(n_features)]).T
            x_minus_j = np.array([z[:, col] if col in z_idx + [j] else x[:, col] for col in range(n_features)]).T

            x_plus_j_list.append(x_plus_j)
            x_minus_j_list.append(x_minus_j)

        x_plus_j_list = np.stack(x_plus_j_list)
        x_minus_j_list = np.stack(x_minus_j_list)

        # calculate marginal contribution
        marginal_contribution = return_prob(x_plus_j_list, x_year) - return_prob(x_minus_j_list, x_year)

        feature_idxs.append(j)
        feature_idxs.sort()

        phi_j_x = np.nanmean(marginal_contribution)
        shapley_scores.append(phi_j_x)

    shapley_scores = np.array(shapley_scores)
    np.save(f'{output_path}/shapley_values_column_{column_name}.npy', shapley_scores)

comm.Barrier()

if rank == 0:

    os.chdir(output_path)
    df = pd.DataFrame(columns=column_names)

    for alt_col_name, col_name in zip(alternative_col_names, column_names):
        column_data = np.squeeze(np.load(f'./shapley_values_column_{alt_col_name}.npy'))
        df[col_name] = column_data

    df.index = test_dates
    if AGGREGATE_LOCATIONS:
        df.to_pickle('../temporal_shapley_values.pkl')
    else:
        location_to_operate_on = location_to_operate_on.replace(' ', '_')
        df.to_pickle(f'../temporal_shapley_values_{location_to_operate_on}.pkl')

    stop = time.time()
    complete_time = round((stop - start)/3600, 3)
    print('TOOK:', complete_time, 'HOURS TO COMPLETE')



