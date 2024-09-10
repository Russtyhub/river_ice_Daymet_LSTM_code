#!/usr/bin/python3
# conda activate DL

# This step makes a forecast on the breakup date for each year iteratively (by year)
# It then adds the information from that year back into the validation dataset, gives
# the LSTM a chance to retrain the model on the latest information for 10 epochs
# with a patience of 5, then assesses the next year. Predictions are saved as a numpy 
# file to: predictions_iteratively_main_{number_of_locations_in_main_df}_locations.npy

import os
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import tensorflow as tf
import numpy as np
import pandas as pd

#################################################
# Script Parameters:
lstm_model_path = '/mnt/locutus/remotesensing/r62/river_ice_breakup/Results/DAYMET_Results/Models/DAYMET/DAYMET_BAYESIAN'
path_to_functions = '/home/r62/repos/russ_repos/Functions'
final_datasets_path = '/mnt/locutus/remotesensing/r62/river_ice_breakup/final_Daymet_datasets'
temp_dir_path = '/mnt/locutus/remotesensing/r62/river_ice_breakup/temp'
number_of_locations_in_main_df = 20
BATCH_SIZE = 1200
LOOKBACK_WINDOW = 457
EPOCHS = 10
PATIENCE = 5
METRIC = 'AUC'
DIRECTION = 'max'
RESULTS_PATH = '/mnt/locutus/remotesensing/r62/river_ice_breakup/Results/DAYMET_Results'
main_results_path = '/mnt/locutus/remotesensing/r62/river_ice_breakup/Results/DAYMET_Results/Output/Main'
#################################################

sys.path.append(path_to_functions)

# from STANDARD_FUNCTIONS import read_pickle
from TF_FUNCTIONS import df_to_LSTM, load_model

if not os.path.exists(temp_dir_path):
    os.mkdir(temp_dir_path)

pkl_data = pd.read_pickle(f'{final_datasets_path}/DAYMET_{number_of_locations_in_main_df}_LOCATIONS_PRE_LSTM.pkl')
df = pkl_data['df']
train_y = pkl_data['train_y']
val_y = pkl_data['val_y']
test_y = pkl_data['test_y']
y = np.concatenate([train_y, val_y, test_y])

del train_y, val_y, test_y

model = load_model(lstm_model_path)

# This is where the temporary directory is on Theseus
# /mnt/locutus/remotesensing/r62/river_ice_breakup/temp
# training [1980 - 2003]
# validation [2004 - 2013]
# testing [2014 - 2022]

# As shown above the last year of training is 2003
last_year_of_training = 2003

def generator(mmap_arr_X, mmap_arr_y, batch_size):
    while True:
        splits = np.ceil(mmap_arr_X.shape[0]/batch_size)
        split_X = np.array_split(mmap_arr_X, splits, axis = 0)
        split_y = np.array_split(mmap_arr_y, splits, axis = 0)

        for X, y in zip(split_X, split_y):
            yield X, np.expand_dims(y, axis=-1)

def delete_directory_contents(directory_path, delete_directory = False):
    if os.path.exists(directory_path):
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # Remove the file or link
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    if delete_directory:
        os.rmdir(directory_path)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'{RESULTS_PATH}/Models/temp_iterative_evaluation/CHECKPOINTS/model.' + '{epoch:02d}.keras',
    save_weights_only=False,
    monitor=f'val_{METRIC}',
    mode=DIRECTION,
    save_best_only=False)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor=f'val_{METRIC}', # early stopping doesn't work with f1_score 
    verbose=1,
    patience=PATIENCE,
    mode=DIRECTION,
    restore_best_weights=True)

LSTM_data = df_to_LSTM(df, LOOKBACK_WINDOW)
df = df.iloc[LOOKBACK_WINDOW - 1:, :]
y = y[LOOKBACK_WINDOW - 1:]


predictions = []
delete_directory_contents(f'{RESULTS_PATH}/Models/temp_iterative_evaluation/CHECKPOINTS/')

for idx, year in enumerate(range(2014, 2023+1)):

    print(f'Evaluating Year: {year}')

    # setting the training data
    train_mask = (df.index.year <= last_year_of_training + idx)
    training = LSTM_data[train_mask]
    train_y = y[train_mask]
    del train_mask

    # Overwrite the above training in mmap_mode
    np.save(f'{temp_dir_path}/training.npy', training)
    np.save(f'{temp_dir_path}/train_y.npy', train_y)
    training = np.load(f'{temp_dir_path}/training.npy', mmap_mode='r')
    train_y = np.load(f'{temp_dir_path}/train_y.npy', mmap_mode='r')

    # setting the validation data
    val_mask = (df.index.year >= last_year_of_training + 1 + idx) & (df.index.year <= year - 1)
    validation = LSTM_data[val_mask]
    val_y = y[val_mask]
    del val_mask

    # Overwrite the above validation in mmap_mode
    np.save(f'{temp_dir_path}/validation.npy', validation)
    np.save('/mnt/locutus/remotesensing/r62/river_ice_breakup/temp/val_y.npy', val_y)
    validation = np.load(f'{temp_dir_path}/validation.npy', mmap_mode='r')
    val_y = np.load(f'{temp_dir_path}/val_y.npy', mmap_mode='r')

    # Setting the testing data (not in mmap mode as its already pretty small)
    test_mask = (df.index.year == year)
    testing = LSTM_data[test_mask]
    del test_mask 

    if idx == 0:
        pass
    else:
        hist = model.fit(generator(training, train_y, BATCH_SIZE),
                  validation_data = generator(validation, val_y, BATCH_SIZE),
                  steps_per_epoch = int(np.ceil(len(training)/BATCH_SIZE)),
                  validation_steps = int(np.ceil(len(validation)/BATCH_SIZE)),
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  callbacks = [model_checkpoint, early_stopping])
        
        val_METRIC_per_epoch = hist.history[f'val_{METRIC}']
        best_epoch = val_METRIC_per_epoch.index(max(val_METRIC_per_epoch)) + 1
        model.load_weights(f'{RESULTS_PATH}/Models/temp_iterative_evaluation/CHECKPOINTS/model.' + f'{best_epoch:02d}.keras')

    preds = np.squeeze(model.predict(testing))
    predictions.append(preds)

    # save the model for each iteration
    model_path = f'{RESULTS_PATH}/Models/temp_iterative_evaluation/iterative_models/best_model_{year}.keras'
    model.save(model_path)
    delete_directory_contents(f'{RESULTS_PATH}/Models/temp_iterative_evaluation/CHECKPOINTS/')

# delete_directory_contents(temp_dir_path, delete_directory=True)
predictions = np.concatenate(predictions)
np.save(f'{main_results_path}/predictions_iteratively_main_{number_of_locations_in_main_df}_locations.npy', predictions)











