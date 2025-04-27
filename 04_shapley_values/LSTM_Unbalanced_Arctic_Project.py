#!/usr/bin/python3

# conda activate DL
# python3 LSTM_Unbalanced_Arctic_Project.py < MODEL > < VERSION > < TUNE > < RUN_TITLE >

# For Example: python3 LSTM_Unbalanced_Arctic_Project.py CanESM5 HISTORICAL False example_run
# If you are doing a Daymet run, MODEL and VERSION should each just be DAYMET

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import datetime
import numpy as np
import pandas as pd
import time as TM
import copy
import sys
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

####################################  PARAMETERS #############################################################

SEED = 123
EPOCHS = (25, 12, 100)
WINDOW_SIZE = 457 # 365 + days in the April + May + June (+1)
BATCH_SIZE = WINDOW_SIZE*16
TUNER = 'BAYESIAN'
NUMBER_OF_BAYESIAN_TRIALS = 70
OVERWRITE_TRIALS = False
PATIENCE = 5
METRIC = 'AUC'
TRAIN = True

####################################  FUNCTIONS  ##############################################################

sys.path.append('/home/r62/repos/russ_repos/Functions/')
sys.path.append('/home/r62/repos/russ_repos/DL_Models/')

from DATA_ANALYSIS_FUNCTIONS import normalize_df, split_train_val_test
from TF_FUNCTIONS import df_to_LSTM, tf_set_seeds, convert_to_TF_data_obj, load_model
from STANDARD_FUNCTIONS import write_pickle, read_pickle, mask_df_to_x, remove_folder_contents, create_directory

from LSTM_Transformer import LSTM_Transformer

############################ SETTING THE ENVIRONMENT #######################################################

MODEL = sys.argv[-4]
version = sys.argv[-3].upper()
TUNE = sys.argv[-2].upper()
RUN_TITLE = sys.argv[-1].upper()

RESULTS_PATH = f'/mnt/locutus/remotesensing/r62/river_ice_breakup/Results/{RUN_TITLE}'

if TUNE == 'FALSE':
	TUNE = False
elif TUNE == 'TRUE':
	TUNE = True
else:
    raise Exception('PLEASES SPECIFY TRUE OR FALSE FOR TUNE')
	
if version.upper() == 'FUTURE':
	TUNE = False

if version.upper() == 'DAYMET':
	MODEL = 'DAYMET'

if TUNE and OVERWRITE_TRIALS:
	print('ARE YOU SURE YOU WANT TO OVERWRITE THE CURRENT TRIALS? (yes/no)')
	x = input()
	if (x.upper() == 'YES') and (os.path.exists(f'{RESULTS_PATH}/Tuning/TUNING_LSTM_{MODEL}_{TUNER}/')):
		remove_folder_contents(f'{RESULTS_PATH}/Tuning/TUNING_LSTM_{MODEL}_{TUNER}/')
	elif x.upper() == 'NO':
		raise Exception('PROJECT HAULTED')
	else:
		raise Exception('PROJECT HAULTED YOU DID NOT SAY YES OR NO')

# tf.debugging.set_log_device_placement(True) # this turns the output into a mess! (might sometimes be useful however)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')

print("Number of GPUs Available: ", len(gpus), flush = True)
print("Number of CPUs Available: ", len(cpus), flush = True)

strategy_gpus = tf.distribute.MirroredStrategy(devices = [f"/GPU:{i}" for i in range(len(gpus))]) # runs even if there are no GPUs (just a useless object on theseus)
strategy_cpus = tf.distribute.MirroredStrategy(devices = [f"/CPU:{i}" for i in range(len(cpus))])
strategy = tf.distribute.MirroredStrategy()

METRIC = METRIC.upper()

################################### SETTING SEED & SCALING BATCH SIZE ##################################

tf_set_seeds(SEED)
pkl_dict = {}

if len(gpus) > 0:
	BATCH_SIZE = BATCH_SIZE*strategy_gpus.num_replicas_in_sync
	print('EFFECTIVE BATCH SIZE:', BATCH_SIZE)
else:
	print('EFFECTIVE BATCH SIZE:', BATCH_SIZE)

# if len(gpus) > 0:
# 	modeling_strategy = strategy_gpus
# else:
# 	modeling_strategy = strategy_cpus

METRICS = [tf.keras.metrics.Precision(name='PRECISION'),
		   tf.keras.metrics.Recall(name='RECALL'),
		   tf.keras.metrics.AUC(name='AUC'),]

if METRIC in ['AUC', 'RECALL', 'PRECISION', 'ACCURACY']:
	DIRECTION = 'max'

##################################### ENSURING IMPORT DATA AND FILE PATHS EXIST ################################

if version == 'DAYMET':
	DAYMET_LSTM_DATA_PATH = '/mnt/locutus/remotesensing/r62/river_ice_breakup/final_inputs/DAYMET_25_LOCATIONS_PRE_LSTM.pkl'
	if os.path.exists(DAYMET_LSTM_DATA_PATH):
		pass
	else:
		raise Exception("IMPORT DATA FOR DAYMET DOES NOT EXIST AT THAT FILE PATH")
		
elif version == 'FUTURE':
	HISTORICAL_LSTM_DATA_PATH = f'/mnt/locutus/remotesensing/r62/river_ice_breakup/CMIP6/Processed/{MODEL}/NORMED_PRE_LSTM_{MODEL}_with_MASK_HISTORICAL.pkl'
	ens = ['SSP119', 'SSP245', 'SSP370', 'SSP585', 'SSP534-OVER']
	for e in ens:
		if os.path.exists(f'/mnt/locutus/remotesensing/r62/river_ice_breakup/CMIP6/Processed/{MODEL}/NORMED_PRE_LSTM_{MODEL}_{e}.pkl'):
			pass
		else:
			print(f'RUNNING FOR FUTURE SIMULATIONS, EXPERIMENT: {e} MODEL: {MODEL} DOES NOT EXIST AT THAT FILE PATH')

elif version == 'HISTORICAL':
	HISTORICAL_LSTM_DATA_PATH = f'/mnt/locutus/remotesensing/r62/river_ice_breakup/CMIP6/Processed/{MODEL}/NORMED_PRE_LSTM_{MODEL}_with_MASK_HISTORICAL.pkl'
	if os.path.exists(HISTORICAL_LSTM_DATA_PATH):
		pass
	else:
		print(f"IMPORT DATA FOR HISTORICAL MODEL: {MODEL} DOES NOT EXIST AT THAT FILE PATH")

paths_to_create = [f'{RESULTS_PATH}/Output/{MODEL}/',
				   f'{RESULTS_PATH}/Models/',
				   f'{RESULTS_PATH}/Tuning/',
				   f'{RESULTS_PATH}/Models/CHECKPOINTS/{MODEL}/']
		
create_directory(paths_to_create)
	
####################################### ALLOCATING DATA #########################################################

if version == 'DAYMET':
	
	# with strategy_cpus.scope(): #  tf.device('/CPU:0') # add a tab to everything till elif version == 'FUTURE': if you want to
	# use this strategy for memory allocation
		
	data = read_pickle(DAYMET_LSTM_DATA_PATH)

	train_df = data['train_df']
	val_df = data['val_df']
	test_df = data['test_df']

	train_y = data['train_y']
	val_y = data['val_y']
	test_y = data['test_y']

	actuals_train_val = np.concatenate([train_y, val_y])

	neg = len(actuals_train_val[actuals_train_val == 0])
	pos = len(actuals_train_val[actuals_train_val == 1])
	initial_bias = np.log([pos/neg])

	train_y = np.array(train_y)[WINDOW_SIZE:]
	val_y = np.array(val_y)[WINDOW_SIZE:]
	test_y = np.array(test_y)[WINDOW_SIZE:]

	# reshape the y variables:
	# train_y = np.reshape(train_y, (-1, 1))
	# val_y = np.reshape(val_y, (-1, 1))
	# test_y = np.reshape(test_y, (-1, 1))
		
	train_X = df_to_LSTM(train_df, window_size=WINDOW_SIZE)
	val_X = df_to_LSTM(val_df, window_size=WINDOW_SIZE)
	test_X = df_to_LSTM(test_df, window_size=WINDOW_SIZE)
	number_of_features = train_df.shape[1]

	del train_df, val_df, test_df, actuals_train_val

	train_data = convert_to_TF_data_obj((train_X, train_y), BATCH_SIZE)
	del train_X, train_y

	val_data = convert_to_TF_data_obj((val_X, val_y), BATCH_SIZE)
	del val_X, val_y

	test_data = convert_to_TF_data_obj((test_X, test_y), BATCH_SIZE)
	del test_X, test_y
		
		
elif version == 'FUTURE':
	
	PERC_TRAIN = 0.7
	
	# with strategy_cpus.scope():

	pkl_data = pd.read_pickle(HISTORICAL_LSTM_DATA_PATH)
	data = pkl_data['data']
	mask = np.concatenate(pkl_data['mask'])
	actuals = data.pop('Breakup Date')

	data = mask_df_to_x(data, mask, -1)

	train_df = data.iloc[0:int(len(data)*PERC_TRAIN), :]
	val_df = data.iloc[int(len(data)*(PERC_TRAIN)):, :]
	number_of_features = train_df.shape[1]

	train_y = actuals.iloc[0:int(len(data)*PERC_TRAIN)]
	val_y = actuals.iloc[int(len(data)*(PERC_TRAIN)):]

	neg = len(actuals[actuals == 0])
	pos = len(actuals[actuals == 1])
	initial_bias = np.log([pos/neg])

	pkl_dict['train_df'] = train_df
	pkl_dict['val_df'] = val_df

	pkl_dict['train_y'] = train_y
	pkl_dict['val_y'] = val_y

	train_y = train_y[WINDOW_SIZE:]
	val_y = val_y[WINDOW_SIZE:]
	
	# reshape the y variables:
	# train_y = np.reshape(train_y, (-1, 1))
	# val_y = np.reshape(val_y, (-1, 1))

	train_X = df_to_LSTM(train_df, window_size=WINDOW_SIZE)
	val_X = df_to_LSTM(val_df, window_size=WINDOW_SIZE)

	del train_df, val_df, actuals

	train_data = convert_to_TF_data_obj((train_X, train_y), BATCH_SIZE)
	del train_X, train_y

	val_data = convert_to_TF_data_obj((val_X, val_y), BATCH_SIZE)
	del val_X, val_y
								
elif version == 'HISTORICAL':
	PERC_TRAIN = 0.6
	PERC_VAL = 0.2
	
	# with strategy_cpus.scope():
				
	pkl_data = pd.read_pickle(HISTORICAL_LSTM_DATA_PATH)
	data = pkl_data['data']
	mask = np.concatenate(pkl_data['mask'])
	actuals = data.pop('Breakup Date')

	train_df, val_df, test_df = split_train_val_test(data, PERC_TRAIN, PERC_VAL, how='sequential')
	train_mask, val_mask, _ = split_train_val_test(mask, PERC_TRAIN, PERC_VAL, how='sequential')
	train_y, val_y, test_y = split_train_val_test(actuals, PERC_TRAIN, PERC_VAL, how='sequential')

	number_of_features = train_df.shape[1]

	train_df = mask_df_to_x(train_df, train_mask, -1)
	val_df = mask_df_to_x(val_df, val_mask, -1)	

	val_train_actuals = np.concatenate([train_y, val_y], axis = 0)		

	neg = len(val_train_actuals[val_train_actuals == 0])
	pos = len(val_train_actuals[val_train_actuals == 1])
	initial_bias = np.log([pos/neg])

	pkl_dict['train_df'] = train_df
	pkl_dict['val_df'] = val_df
	pkl_dict['test_df'] = test_df

	pkl_dict['train_y'] = train_y
	pkl_dict['val_y'] = val_y
	pkl_dict['test_y'] = test_y

	train_y = train_y[WINDOW_SIZE:]
	val_y = val_y[WINDOW_SIZE:]
	test_y = test_y[WINDOW_SIZE:]

	# reshape the y variables:
	# train_y = np.reshape(train_y, (-1, 1))
	# val_y = np.reshape(val_y, (-1, 1))
	# test_y = np.reshape(test_y, (-1, 1))
	
	train_X = df_to_LSTM(train_df, window_size=WINDOW_SIZE)
	val_X = df_to_LSTM(val_df, window_size=WINDOW_SIZE)
	test_X = df_to_LSTM(test_df, window_size=WINDOW_SIZE)

	del train_df, val_df, test_df, actuals

	train_data = convert_to_TF_data_obj((train_X, train_y), BATCH_SIZE)
	del train_X, train_y

	val_data = convert_to_TF_data_obj((val_X, val_y), BATCH_SIZE)
	del val_X, val_y

	test_data = convert_to_TF_data_obj((test_X, test_y), BATCH_SIZE)
	del test_X, test_y

start = TM.time()

# with strategy.scope():


hyper_transformer = LSTM_Transformer(input_shape = (WINDOW_SIZE, number_of_features),
                         optimizer='adam',
                         loss='BCE',
                         metrics = METRICS, 
                         momentum=None,
                         initial_bias=initial_bias,
                         mask_value = -1,
						 reduce_transformer_window=True)

########################################### CALLBACKS ############################################################

early_stopping = tf.keras.callbacks.EarlyStopping(
	monitor=f'val_{METRIC}', # early stopping doesn't work with f1_score 
	verbose=1,
	patience=PATIENCE,
	mode=DIRECTION,
	restore_best_weights=True)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
	filepath=f'{RESULTS_PATH}/Models/CHECKPOINTS/{MODEL}/{MODEL}_{version}_{TUNER}/',
	save_weights_only=False,
	monitor=f'val_{METRIC}',
	mode=DIRECTION,
	save_best_only=True)

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

############################### MODEL TUNING #######################################################################

if TUNER == 'HYPERBAND':
# Hyperband determines the number of models to train in a bracket by computing 1 + log_factor(max_epochs) and rounding it up to the nearest integer.

	tuner = kt.Hyperband(hyper_transformer,
						 objective = kt.Objective(f'val_{METRIC}', direction=DIRECTION),
						 max_epochs=EPOCHS[1],
						 overwrite=False,
						 factor=3,
						 directory = f'{RESULTS_PATH}/Tuning/',
						 project_name = f'TUNING_LSTM_{MODEL}_HYPERBAND/',)
						 # distribution_strategy = strategy)

elif TUNER == 'BAYESIAN':
	tuner = kt.BayesianOptimization(hyper_transformer,
						 objective = kt.Objective(f'val_{METRIC}', direction=DIRECTION), 
						 max_trials = NUMBER_OF_BAYESIAN_TRIALS,
						 overwrite=False,
						 directory = f'{RESULTS_PATH}/Tuning/',
						 project_name = f'TUNING_LSTM_{MODEL}_BAYESIAN/',)
						 # distribution_strategy = strategy)
if TUNE:
	print('TUNING IN PROGRESS')
	tuner.overwrite = OVERWRITE_TRIALS
	tf.keras.backend.clear_session()
	tuner.search(train_data, batch_size=BATCH_SIZE, epochs=EPOCHS[0], validation_data=val_data, callbacks=[early_stopping, end_loss_equal_to_nan])
	print('TUNING COMPLETE')

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0] # Get the optimal hyperparameters
model = tuner.hypermodel.build(best_hps)

########################################### MODEL TRAINING ###############################################################

if os.path.exists(f'{RESULTS_PATH}/Models/{MODEL}/LSTM_{TUNER}_{MODEL}_{version}') and TRAIN == False:
	hypermodel = load_model(f'{RESULTS_PATH}/Models/LSTM_{TUNER}_{MODEL}_{version}')

else:
	early_stopping.patience = PATIENCE + 20
	hist = model.fit(train_data, batch_size=BATCH_SIZE, epochs=EPOCHS[2], validation_data=val_data,
					 callbacks=[early_stopping, model_checkpoint, end_loss_equal_to_nan, reduce_learning_rate])
	val_METRIC_per_epoch = hist.history[f'val_{METRIC}']
	best_epoch = val_METRIC_per_epoch.index(max(val_METRIC_per_epoch)) + 1
	print(f'BEST EPOCH: {best_epoch}')

hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.load_weights(f'{RESULTS_PATH}/Models/CHECKPOINTS/{MODEL}/{MODEL}_{version}_{TUNER}/')
hypermodel.save(f'{RESULTS_PATH}/Models/{version.upper()}_{TUNER}') # .hdf5 format created errors

######################################### SAVING MODEL AND OUTPUTS ########################################################

if version == 'DAYMET':
	pkl_dict['predicted_probs'] = np.squeeze(hypermodel.predict(test_data))
	hypermodel.save(f'{RESULTS_PATH}/Models/DAYMET/LSTM_{TUNER}_{version}') 

elif version == 'FUTURE':
	for e in ens:
		try:
			ssp = pd.read_pickle(f'/mnt/locutus/remotesensing/r62/river_ice_breakup/CMIP6/Processed/{MODEL}/NORMED_PRE_LSTM_{MODEL}_{e}.pkl').iloc[:, 1:]
			ssp_X = df_to_LSTM(ssp, window_size=WINDOW_SIZE)
			ssp_X = convert_to_TF_data_obj(ssp_X, BATCH_SIZE)
			print(f'RUNNING SIMULATION FOR MODEL: {MODEL} EXPERIMENT: {e}')
			pkl_dict[f'predicted_probs_{e}'] = np.squeeze(hypermodel.predict(ssp_X))
		except:
			print(f'EXPERIMENT {e} DOES NOT EXIST')
			
elif version == 'HISTORICAL':
	pkl_dict['predicted_probs'] = np.squeeze(hypermodel.predict(test_data))
	hypermodel.save(f'{RESULTS_PATH}/Models/{MODEL}/LSTM_{TUNER}_{MODEL}_{version}') 
	
write_pickle(f'{RESULTS_PATH}/Output/{MODEL}/OUTPUT_LSTM_{MODEL}_{TUNER}_{version}', pkl_dict)

#############################################################################################################################

stop = TM.time()
complete = (stop - start)/3600

print('Process complete! Took ', round(complete, 2), 'hours', flush = True)


