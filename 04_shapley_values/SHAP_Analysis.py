#!/usr/bin/python3
# conda activate DL

import pandas as pd
import numpy as np
import shap
import os
import sys
import time as TM
import tensorflow as tf

sys.path.append('../../Functions')
from STANDARD_FUNCTIONS import read_pickle
from TF_FUNCTIONS import load_model, df_to_LSTM, convert_to_TF_data_obj

WINDOW_SIZE = 457
BATCH_SIZE = WINDOW_SIZE*16
model = load_model('/mnt/locutus/remotesensing/r62/river_ice_breakup/Results/LOCATIONS_25/Models/DAYMET/DAYMET_BAYESIAN')

input_data = read_pickle('/mnt/locutus/remotesensing/r62/river_ice_breakup/final_inputs/DAYMET_25_LOCATIONS_PRE_LSTM.pkl')
test_locations = input_data['test_location'] 
test_dates = input_data['test_dates'] 
test_df = input_data['test_df']
test_X = df_to_LSTM(test_df, window_size=WINDOW_SIZE)
test_y = input_data['test_y'][WINDOW_SIZE:]

# test_data = convert_to_TF_data_obj((test_X, test_y), BATCH_SIZE)

predictions = read_pickle('/mnt/locutus/remotesensing/r62/river_ice_breakup/Results/LOCATIONS_25/Output/DAYMET/OUTPUT_LSTM_DAYMET_BAYESIAN_DAYMET')['predicted_probs']

# print(test_df.shape, predictions.shape)
print(model.summary())

# Creates a wrapper function and then the SHAP explainer:
def f(X):
    return model.predict([X[:, i] for i in range(X.shape[-1])]).flatten()

explainer = shap.KernelExplainer(f, test_X,) # max_evals=1000

# Create a sample from test (recall test values are not masked):
sample_idx = np.random.choice(test_X.shape[0], 500, replace = False) # should come from the training but...
sample = test_X[sample_idx]
del sample_idx

# Calculates the SHAP values - It takes some time
print('CREATING SHAP VALUES')

start = TM.time()
shap_values = explainer.shap_values(sample)
# shap_values = explainer(sample)

# save the shap values
np.save("/mnt/locutus/remotesensing/r62/river_ice_breakup/Results/LOCATIONS_25/Output/DAYMET/shap_values.npy", shap_values)

stop = TM.time()
complete = (stop - start)/3600

print('Process complete! Took ', round(complete, 2), 'hours', flush = True)
