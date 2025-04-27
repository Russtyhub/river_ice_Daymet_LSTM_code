#!/usr/bin/python3
# conda activate DL
# python3 01_iterative_LSTM_evaluation.py

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
sys.path.append('../')
from resources import df_to_LSTM

#################################################
# Send to where you house your data
results_path = '/mnt/locutus/remotesensing/r62/river_ice_breakup/Results/DAYMET_Results'
lstm_model_path = f'{results_path}/Models/DAYMET/DAYMET_BAYESIAN'
final_datasets_path = '/mnt/locutus/remotesensing/r62/river_ice_breakup/final_Daymet_datasets'
number_of_locations_in_main_df = 23
BATCH_SIZE = 1200
LOOKBACK_WINDOW = 457
EPOCHS = 25
PATIENCE = 7
METRIC = 'AUC'
DIRECTION = 'max'
#################################################

def load_model(path):
    return tf.keras.models.load_model(path)

pkl_data = pd.read_pickle(f'{final_datasets_path}/DAYMET_{number_of_locations_in_main_df}_LOCATIONS_PRE_LSTM.pkl')
df = pkl_data['df']
train_y = pkl_data['train_y']
val_y = pkl_data['val_y']
test_y = pkl_data['test_y']
y = np.concatenate([train_y, val_y, test_y])

del train_y, val_y, test_y

model = load_model(lstm_model_path)

# This is where the temporary directory is on Theseus
# training [1980 - 2003]
# validation [2004 - 2013]
# testing [2014 - 2022]

# As shown above the last year of training is 2003
last_year_of_training = 2003

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
    filepath=f'{results_path}/Models/temp_iterative_evaluation/CHECKPOINTS/model.' + '{epoch:02d}.keras',
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
delete_directory_contents(f'{results_path}/Models/temp_iterative_evaluation/CHECKPOINTS/')

print('*'*40)
for idx, year in enumerate(range(2014, 2023+1)):

    print(f'Evaluating Year: {year}')

    # setting the training data
    train_mask = (df.index.year <= last_year_of_training + idx)
    training = LSTM_data[train_mask]
    train_y = y[train_mask]
    del train_mask

    # sanity check
    print('First year of training:', df.index.year.min())
    print('Last year of training:', last_year_of_training + idx)
    print('training shape:', training.shape)

    # setting the validation data
    val_mask = (df.index.year >= last_year_of_training + 1 + idx) & (df.index.year <= year - 1)
    validation = LSTM_data[val_mask]
    val_y = y[val_mask]
    del val_mask

    # sanity check
    print('First year of validation:', last_year_of_training + 1 + idx)
    print('Last year of validation:', year - 1)
    print('validation shape:', validation.shape)

    # setting the bias:
    actuals_train_val = np.concatenate([train_y, val_y])

    neg = len(actuals_train_val[actuals_train_val == 0])
    pos = len(actuals_train_val[actuals_train_val == 1])
    initial_bias = np.log([pos/neg])
    del actuals_train_val

    # Setting the testing data
    test_mask = (df.index.year == year)
    testing = LSTM_data[test_mask]
    print('testing shape:', testing.shape)
    print('*'*40)

    del test_mask 

    if idx > 0:
       
        # updating the output layer bias:
        output_layer = model.layers[-1]
        output_layer.bias.assign(initial_bias)

        hist = model.fit(training, train_y,
                  validation_data = (validation, val_y),
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  callbacks = [model_checkpoint, early_stopping])
        
        val_METRIC_per_epoch = hist.history[f'val_{METRIC}']
        best_epoch = val_METRIC_per_epoch.index(max(val_METRIC_per_epoch)) + 1
        model.load_weights(f'{results_path}/Models/temp_iterative_evaluation/CHECKPOINTS/model.' + f'{best_epoch:02d}.keras')

    preds = np.squeeze(model.predict(testing))
    np.save(f'{results_path}/Output/Main/annual_iterative_predictions/predictions_{year}.npy', preds)
    predictions.append(preds)

    # save the model for each iteration
    model_path = f'{results_path}/Models/temp_iterative_evaluation/iterative_models/best_model_{year}.keras'
    model.save(model_path)
    delete_directory_contents(f'{results_path}/Models/temp_iterative_evaluation/CHECKPOINTS/')

predictions = np.concatenate(predictions)
np.save(f'{results_path}/Output/Main/predictions_iteratively_main_{number_of_locations_in_main_df}_locations.npy', predictions)







