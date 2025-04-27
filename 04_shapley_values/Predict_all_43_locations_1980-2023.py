#!/usr/bin/python3

# Now use DL environment
import pandas as pd
import numpy as np
import sys

sys.path.append('/home/r62/repos/russ_repos/Functions/')
from TF_FUNCTIONS import load_model, df_to_LSTM
model = load_model('/mnt/locutus/remotesensing/r62/river_ice_breakup/Results/DAYMET_25_LOCATIONS/Models/DAYMET/DAYMET_BAYESIAN')

data = pd.read_pickle('/mnt/locutus/remotesensing/r62/river_ice_breakup/final_inputs/impute_missing_43_locations_at_least_10_1980-2023.pkl')
data = df_to_LSTM(data, 457)

predictions = model.predict(data)
predictions = np.squeeze(predictions)

np.save('/mnt/locutus/remotesensing/r62/river_ice_breakup/Results/DAYMET_25_LOCATIONS/Output/DAYMET/predictions_for_43_locations_1980-2023.npy', predictions)