#!/usr/bin/python3
# Perlmutter: module load tensorflow/2.15.0

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
from mpi4py import MPI
import pickle

sys.path.append('../')

from resources import check_trial_files, df_to_LSTM, Slurm_info

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

def load_model(path):
    return tf.keras.models.load_model(path)

####################################  PARAMETERS ############################################################# 

EPOCHS = (12, 200)
TUNER = 'BAYESIAN'
WINDOW_SIZE = 457
NUMBER_OF_BAYESIAN_TRIALS = 100
PATIENCE = 20
METRIC = 'AUC'
CARE_IF_ALL_TRIALS_RAN = False
WHICH_MODEL = 'BUILD_HERE' # BEST_TUNED, BUILD_HERE, OTHER
HPC = True
############################ SETTING THE ENVIRONMENT #######################################################

tf_set_seeds(SEED)
slurm_info = Slurm_info()

#synchronize all nodes:
comm = MPI.COMM_WORLD
comm.Barrier()

if HPC:
    slurm_rank = int(os.environ['SLURM_PROCID'])
else:
    slurm_rank = 0

if HPC:
    if slurm_rank == 0:
        JOB_TITLE = 'chief'
        port_number = 8008
        nodes = slurm_info.nodes
        chief_node = os.environ.get('SLURMD_NODENAME')
        nodes.remove(chief_node)
        chief_node = [chief_node]
        chief_node_json = json.dumps(chief_node)

        nodes_with_port_numbers = []
        for idx, i in enumerate(nodes, start = 1):
            p_number = idx*5 + port_number
            val = f'{i}:{p_number}'
            nodes_with_port_numbers.append(val)
        nodes_with_port_numbers_json = json.dumps(nodes_with_port_numbers)
        print('CHIEF NODE:', chief_node[0] + ':8008', '\nWORKER NODES:', nodes)

    else:
        JOB_TITLE = 'worker'
        nodes_with_port_numbers_json = None
        chief_node_json = None

    nodes_with_port_numbers_json = comm.bcast(nodes_with_port_numbers_json, root=0)
    nodes_with_port_numbers = json.loads(nodes_with_port_numbers_json)

    chief_node_json = comm.bcast(chief_node_json, root=0)
    chief_node = json.loads(chief_node_json)[0]

    # there should be a worker with an index of 0 and the chief 
    # gets an index of 0 as well
    work_id = slurm_rank - 1
    if work_id < 0:
        work_id = 0

    os.environ["TF_CONFIG"] = json.dumps({
       "cluster": {
           "chief": [f"{chief_node}:8008"],
           "worker": nodes_with_port_numbers,
       },
      "task": {"type": JOB_TITLE, "index": work_id}
    })   

    # Set the TF_CONFIG environment variable to configure the cluster setting. 
    # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ["TF_CUDNN_USE_AUTOTUNE"] = "1"
    # tf.config.run_functions_eagerly(True)
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    # tf.debugging.set_log_device_placement(True) # this turns the output into a mess! (might sometimes be useful however)

#Enable TF-AMP graph rewrite: 
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
#Enable Automated Mixed Precision: 
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = "1"
os.environ['TF_KERAS'] = '1'
    
gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')

if slurm_rank == 0:
    print("Number of GPUs Available per node: ", len(gpus), flush = True)
    print("Number of CPUs Available per node: ", len(cpus), flush = True)

    print('GPUs:', gpus, flush = True)
    print('CPUs:', cpus, flush = True)
    
# I must make the batch size divisible by the number of replicas (GPUs)
# If I try tuning on many CPUs in parallel this should be adjusted!
# GLOBAL_BATCH_SIZE = BATCH_SIZE*len(gpus)

if HPC:
    strategy = tf.distribute.MirroredStrategy()
    n_gpus = len(gpus)
else:
    n_gpus = 1

# num_devices = strategy.num_replicas_in_sync
# print(f'Number of devices: {num_devices}', flush = True)

######################### IMPORTING DATA ####################################################

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

train_y = np.array(train_y)[WINDOW_SIZE:].astype('int32')
val_y = np.array(val_y)[WINDOW_SIZE:].astype('int32')

train_X = df_to_LSTM(train_df, window_size=WINDOW_SIZE)
val_X = df_to_LSTM(val_df, window_size=WINDOW_SIZE)
number_of_features = train_df.shape[1]

train_X = train_X.astype('float32')
val_X = val_X.astype('float32')

del train_df, val_df, actuals_train_val

if slurm_rank == 0:
    print('train_X shape:', train_X.shape, flush = True)
    print('train_y shape:', train_y.shape, flush = True)

    print('val_X shape:', val_X.shape, flush = True)
    print('val_y shape:', val_y.shape, flush = True)

train_n = train_X.shape[0]
val_n = val_X.shape[0]

############################ SETTING UP THE MODEL ###########################################

start = TM.time()

if slurm_rank == 0:
    print(f'BUILDING', DEEP_LEARNING_MODEL, flush = True)
    
METRIC = METRIC.upper()

if strategy:
    with strategy.scope():

        METRIC = METRIC.upper()
        METRICS = [tf.keras.metrics.AUC(name='AUC')]

        model = retrieve_DL_model(DEEP_LEARNING_MODEL)        
        model = model(input_shape = (WINDOW_SIZE, number_of_features),
                            gpus_per_node = n_gpus,
                            optimizer='adam',
                            loss='BCE',
                            max_number_lstm_layers=1,
                            max_number_dense_layers=1,
                            metrics = METRICS, 
                            momentum=None,
                            strategy = strategy)
else:
    
    METRIC = METRIC.upper()
    METRICS = [tf.keras.metrics.AUC(name='AUC')]

    model = retrieve_DL_model(DEEP_LEARNING_MODEL)
    model = model(input_shape = (WINDOW_SIZE, number_of_features),
                            gpus_per_node = n_gpus, 
                            optimizer='adam',
                            loss='BCE',
                            max_number_lstm_layers=1,
                            max_number_dense_layers=1,                         
                            metrics = METRICS, 
                            momentum=None,
                            strategy = None)

DIRECTION = model.produce_direction()

########################################### CALLBACKS ############################################################

if slurm_rank == 0:
    print('SETTING CALLBACKS', flush = True)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor=f'val_{METRIC}', # early stopping doesn't work with f1_score 
    verbose=1,
    patience=PATIENCE,
    mode=DIRECTION,
    restore_best_weights=True)

best_model_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'{results_path}/Models/BEST_MODELS/{DEEP_LEARNING_MODEL}_{TUNER}/model.' + '{epoch:02d}.keras',
    save_weights_only=False,
    monitor=f'val_{METRIC}',
    mode=DIRECTION,
    save_best_only=True)

checkpoint_path = f'{results_path}/Models/CHECKPOINTS/{DEEP_LEARNING_MODEL}_{TUNER}/'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'{results_path}/Models/CHECKPOINTS/{DEEP_LEARNING_MODEL}_{TUNER}/model.' + '{epoch:02d}.weights.h5',
    save_weights_only=True,
    monitor=f'val_{METRIC}',
    mode=DIRECTION,
    save_best_only=False)

end_loss_equal_to_nan = tf.keras.callbacks.TerminateOnNaN()

reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(
    monitor=f'val_{METRIC}',
    factor=0.1,
    patience=PATIENCE,
    verbose=0,
    mode='auto',
    min_delta=0.0001,
    cooldown=0,
    min_lr=0,
)

############################ CHECK TUNER TRIAL DIRS ##############################################
    
if CARE_IF_ALL_TRIALS_RAN:
    trials_dir = f'{results_path}/Tuning/TUNING_{DEEP_LEARNING_MODEL}_{TUNER}'
    all_trials_available = check_trial_files(trials_dir)
    
    if all_trials_available:
        pass
    else:
        raise('ONE OR MORE TRIALS IS MISSING DATA!!')
    
############################ IMPORT MODEL FROM TUNER AND TRIAL OR REBUILD ##########################
    
if WHICH_MODEL.upper() == 'BEST_TUNED':
    
    if slurm_rank == 0:
        print(f'BUILDING {DEEP_LEARNING_MODEL} FROM TUNED HYPERPARAMETERS', flush = True)

    if TUNER == 'HYPERBAND':
    # Hyperband determines the number of models to train in a bracket 
    # by computing 1 + log_factor(max_epochs) and rounding it 
    #up to the nearest integer.

        tuner = kt.Hyperband(model,
                         objective = kt.Objective(f'val_{METRIC}', direction=DIRECTION),
                         max_epochs=EPOCHS[1],
                         overwrite=False,
                         factor=3,
                         directory = f'{results_path}/Tuning/',
                         project_name = f'TUNING_{DEEP_LEARNING_MODEL}_{TUNER}/',
                         distribution_strategy = None)

    elif TUNER == 'BAYESIAN':

        tuner = kt.BayesianOptimization(model,
                         objective = kt.Objective(f'val_{METRIC}', direction=DIRECTION), 
                         max_trials = NUMBER_OF_BAYESIAN_TRIALS,
                         overwrite = False,
                         directory = f'{results_path}/Tuning/',
                         project_name = f'TUNING_{DEEP_LEARNING_MODEL}_{TUNER}/',
                         distribution_strategy = None)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0] # Get the optimal hyperparameters
    model = tuner.hypermodel.build(best_hps)

    if slurm_rank == 0:
        best_trial = tuner.oracle.get_best_trials(num_trials=1)[-1]
        print()
        print(f"Best trial ID: {best_trial.trial_id}")
        print(best_trial.summary())
        print()
    
# I need to create a batch size object here!!!!
    
elif WHICH_MODEL.upper() == 'BUILD_HERE':
    
    if slurm_rank == 0:
        print(f'BUILDING {DEEP_LEARNING_MODEL} USING PARAMETERS PROVIDED', flush = True)
        
    parameters = {'dropout_rate' : 0.0,
                  'recurrent_dropout_rate' : 0.0,
                  'learning_rate' : 0.1,
                  'number_of_lstm_layers' : 1,
                  'number_of_dense_layers' : 1,
                  'batch_size' : WINDOW_SIZE,
                  'dense_layers_activation_funcs' : 'relu',
                  'LSTM_layer_1' : 256,
                  'dense_layer_1' : 256,
                 }
    
    batch_size = parameters['batch_size']
    model = model.build(hp=None, parameters=parameters)

elif WHICH_MODEL.upper() == 'OTHER':
    
    path_to_model = '/path/to/model/to/use.keras'
    
    if slurm_rank == 0:
        print(f'USING THE {DEEP_LEARNING_MODEL} FROM THE PATH YOU PROVIDED:', flush = True)
        print(path_to_model, flush = True)
        
    model = load_model(path_to_model)
    
else:
    raise('SOMETHING IS WRONG WITH THE "WHICH_MODEL" PARAMETER')

if slurm_rank == 0:
    print(model.summary())
    
##################### MODEL TRAINING ###########################################

with strategy.scope():
    
    print('TRAINING IN PROGRESS', flush = True)

    hist = model.fit(x = train_X,
                     y = train_y,
                     validation_data = (val_X, val_y),
                     epochs = EPOCHS[1],
                     batch_size = batch_size*n_gpus,
                     callbacks = [early_stopping,
                                  best_model_callback,
                                  checkpoint_callback,
                                  end_loss_equal_to_nan,
                                  reduce_learning_rate])

    val_METRIC_per_epoch = hist.history[f'val_{METRIC}']

    if DIRECTION.upper() == 'MAX':
        best_epoch = val_METRIC_per_epoch.index(max(val_METRIC_per_epoch)) + 1
    elif DIRECTION.upper() == 'MIN':
        best_epoch = val_METRIC_per_epoch.index(min(val_METRIC_per_epoch)) + 1
    
    if slurm_rank == 0:
        print(f'{DEEP_LEARNING_MODEL}: TRAINING COMPLETE BEST EPOCH: {best_epoch}', flush = True)
           
# saving hist for later:
with open(f'{results_path}/Output/training_history.pkl', 'wb') as f:
    pickle.dump(hist.history, f)
        
stop = TM.time()
complete = (stop - start)/3600
if slurm_rank == 0:
    print('Process complete! Took ', round(complete, 6), 'hours', flush = True)


