#!/usr/bin/python3

# Perlmutter: module load tensorflow/2.15.0
# python3 01_Tuning_LSTM_Perlmutter.py

# info about base tuner in Keras:
# https://keras.io/api/keras_tuner/tuners/base_tuner/

# This script was written for the Perlmutter Supercomputer hosted by NERSC:
# https://docs.nersc.gov/systems/perlmutter/architecture/
# By specifying your desired data and results paths the model should run for you 
# in a similar system using NVIDIA GPUs with CUDA.

# NOTE: Be sure to copy resources.py to the parent directory of this script (../)
# on the system you choose to run the LSTM on

DEEP_LEARNING_MODEL = 'LSTM'
RUN_TITLE = f'{DEEP_LEARNING_MODEL}_01' # to specify which experiment you are running
data_directory = '/path/to/data/directory/on/multicluster/system/'
results_path = f'/path/to/data/directory/on/multicluster/system/{RUN_TITLE}'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import numpy as np
import pandas as pd
import time as TM
import sys
import keras_tuner as kt
import json
import random 
import copy

sys.path.append('../')

from resources import create_directory, delete_everything_in_directory, make_keras_tuner_trials_paths, df_to_LSTM, Slurm_info

SEED = 123
def tf_set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def retrieve_DL_model(deep_learning_model):
    if deep_learning_model.upper() == 'LSTM':
        from LSTM import LSTM
        return LSTM

####################################  PARAMETERS #############################################################

EPOCHS = (20, 12)
WINDOW_SIZE = 457
# BATCH_SIZE = WINDOW_SIZE
TUNER = 'BAYESIAN'
NUMBER_OF_BAYESIAN_TRIALS = 100
OVERWRITE_TRIALS = True
PATIENCE = 6
METRIC = 'AUC'

############################ SETTING THE ENVIRONMENT #######################################################

tf_set_seeds(SEED)
slurm_info = Slurm_info()
slurm_rank = int(os.environ['SLURM_PROCID'])    

if slurm_rank == 0:
    os.environ['KERASTUNER_TUNER_ID'] = 'chief'
else:
    worker_id = slurm_rank - 1
    os.environ['KERASTUNER_TUNER_ID'] = f'tuner{worker_id}'

print(os.environ['KERASTUNER_TUNER_ID'], flush = True)
print('RANK:', slurm_rank, flush = True)
print()

#Enable TF-AMP graph rewrite: 
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
#Enable Automated Mixed Precision: 
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = "1"
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "1"
# tf.config.run_functions_eagerly(True)
os.environ['TF_KERAS'] = "1"
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# tf.debugging.set_log_device_placement(True) # this turns the output into a mess! (might sometimes be useful however)

gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')

if slurm_rank == 0:
    print("Number of GPUs Available per node: ", len(gpus), flush = True)
    print("Number of CPUs Available per node: ", len(cpus), flush = True)

# I must make the batch size divisible by the number of replicas (GPUs)
# If I try tuning on many CPUs in parallel this should be adjusted!
# GLOBAL_BATCH_SIZE = BATCH_SIZE*len(gpus)

print('GPUs:', gpus, flush = True)
print('CPUs:', cpus, flush = True)

strategy = tf.distribute.MirroredStrategy()
# strategy = None

# num_devices = strategy.num_replicas_in_sync
# print(f'Number of devices: {num_devices}', flush = True)

paths_to_create = [f'{results_path}/Output/',
                   f'{results_path}/Models/',
                   f'{results_path}/Tuning/',
                   f'{results_path}/Tuning/TUNING_{DEEP_LEARNING_MODEL}_{TUNER}/',
                   f'{results_path}/Models/CHECKPOINTS/']

# It helps to have the trial dirs created ahead of time for keras tuner in parallel

if OVERWRITE_TRIALS:
    tf.keras.backend.clear_session()
    delete_everything_in_directory(f'{results_path}/Tuning/TUNING_{DEEP_LEARNING_MODEL}_{TUNER}/')

trials_paths = make_keras_tuner_trials_paths(number_of_trials = NUMBER_OF_BAYESIAN_TRIALS,
                                             path = f'{results_path}/Tuning/TUNING_{DEEP_LEARNING_MODEL}_{TUNER}')
    
paths_to_create.extend(trials_paths)
create_directory(paths_to_create, PRINT = False)
    
######################### IMPORTING DATA ###################################################

data = pd.read_pickle(f'{data_directory}/DAYMET_23_LOCATIONS_PRE_LSTM.pkl')

train_df = data['train_df']
val_df = data['val_df']

train_y = data['train_y']
val_y = data['val_y']

actuals_train_val = np.concatenate([train_y, val_y])

neg = len(actuals_train_val[actuals_train_val == 0])
pos = len(actuals_train_val[actuals_train_val == 1])
initial_bias = np.log([pos/neg])

if slurm_rank == 0:
    print('initial_bias:', initial_bias, flush = True)

train_y = np.array(train_y)[WINDOW_SIZE:]
val_y = np.array(val_y)[WINDOW_SIZE:]

# reshape the y variables:
# train_y = np.reshape(train_y, (-1, 1))
# val_y = np.reshape(val_y, (-1, 1))

train_X = df_to_LSTM(train_df, window_size=WINDOW_SIZE)
val_X = df_to_LSTM(val_df, window_size=WINDOW_SIZE)
number_of_features = train_df.shape[1]

del train_df, val_df, actuals_train_val

print('train_X shape:', train_X.shape)
print('train_y shape:', train_y.shape)

print('val_X shape:', val_X.shape)
print('val_y shape:', val_y.shape)


############################ SETTING UP THE MODEL ###########################################

start = TM.time()

if strategy:
    with strategy.scope():

        METRIC = METRIC.upper()
        METRICS = [tf.keras.metrics.AUC(name='AUC')]

        model = retrieve_DL_model(DEEP_LEARNING_MODEL)        
        model = model(input_shape = (WINDOW_SIZE, number_of_features),
                            # batch_size = BATCH_SIZE,
                            gpus_per_node = len(gpus),
                            optimizer='adam',
                            loss='BCE',
                            max_number_lstm_layers=1,
                            max_number_dense_layers=4,
                            metrics = METRICS, 
                            momentum=None,
                            strategy = strategy)
else:
    
    METRIC = METRIC.upper()
    METRICS = [tf.keras.metrics.AUC(name='AUC')]

    model = retrieve_DL_model(DEEP_LEARNING_MODEL)
    model = model(input_shape = (WINDOW_SIZE, number_of_features),
                            # batch_size = BATCH_SIZE,
                            gpus_per_node = len(gpus), 
                            optimizer='adam',
                            loss='BCE',
                            max_number_lstm_layers=1,
                            max_number_dense_layers=4,                         
                            metrics = METRICS, 
                            momentum=None,
                            strategy = None)

DIRECTION = model.produce_direction()

########################################### CALLBACKS ############################################################

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor=f'val_{METRIC}', # early stopping doesn't work with f1_score 
    verbose=1,
    patience=PATIENCE,
    mode=DIRECTION,
    restore_best_weights=True)

end_loss_equal_to_nan = tf.keras.callbacks.TerminateOnNaN()

############################### MODEL TUNING #######################################################################

if TUNER == 'HYPERBAND':
# Hyperband determines the number of models to train in a bracket by computing 
# 1 + log_factor(max_epochs) and rounding it up to the nearest integer.

    tuner = kt.Hyperband(model,
                     objective = kt.Objective(f'val_{METRIC}', direction=DIRECTION),
                     max_epochs = EPOCHS[1],
                     overwrite = False,
                     factor = 3,
                     directory = f'{results_path}/Tuning/',
                     project_name = f'TUNING_{DEEP_LEARNING_MODEL}_{TUNER}/',
                     distribution_strategy = strategy)

elif TUNER == 'BAYESIAN':
    
    tuner = kt.BayesianOptimization(model,
                     objective = kt.Objective(f'val_{METRIC}', direction=DIRECTION), 
                     max_trials = NUMBER_OF_BAYESIAN_TRIALS,
                     overwrite = False,
                     directory = f'{results_path}/Tuning/',
                     project_name = f'TUNING_{DEEP_LEARNING_MODEL}_{TUNER}/',
                     distribution_strategy = strategy)
        
print('TUNING IN PROGRESS', flush = True)
    
tuner.overwrite = OVERWRITE_TRIALS
tf.keras.backend.clear_session()
tuner.search(x = train_X,
             y = train_y,
             validation_data = (val_X, val_y),
             # batch_size=GLOBAL_BATCH_SIZE, 
             epochs=EPOCHS[0],  
             callbacks=[early_stopping, end_loss_equal_to_nan])

print('TUNING COMPLETE', flush = True)

stop = TM.time()
complete = (stop - start)/3600

if slurm_rank == 0:
    print('Process complete! Took ', round(complete, 2), 'hours', flush = True)




