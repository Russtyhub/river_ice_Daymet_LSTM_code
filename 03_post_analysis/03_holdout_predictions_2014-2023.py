#!/usr/bin/python3                                                                              
# conda activate DL
# python3 03_holdout_predictions_2014-2023.py

import sys
import pandas as pd
import numpy as np

sys.path.append('../')

path_to_models = '/mnt/locutus/remotesensing/r62/river_ice_breakup/Results/DAYMET_Results/Models/temp_iterative_evaluation/iterative_models'

from resources import df_to_LSTM

def load_model(path):
    return tf.keras.models.load_model(path)

LOOKBACK_WINDOW = 457

final_datasets_path = '/mnt/locutus/remotesensing/r62/river_ice_breakup/final_Daymet_datasets'
path_to_results = '/mnt/locutus/remotesensing/r62/river_ice_breakup/Results/DAYMET_Results/Output/Holdouts'

pkl_data = pd.read_pickle(f'{final_datasets_path}/DAYMET_10_HOLDOUTS_PRE_LSTM.pkl')
df = pkl_data['df']

actuals = pkl_data['actuals']

LSTM_data = df_to_LSTM(df, LOOKBACK_WINDOW)
df = df.iloc[LOOKBACK_WINDOW - 1:, :]

actuals = actuals[LOOKBACK_WINDOW - 1:]
df['actual'] = actuals

predictions = []
for year in range(2014, 2024):
    
    print('EVALUATING YEAR:', year)
    
    year_mask = df.index.year == year
    temp_df = df.loc[year_mask]
    
    model = load_model(f'{path_to_models}/best_model_{year}.keras')
    preds = model.predict(LSTM_data[year_mask])	
    predictions.append(preds)

predictions = np.concatenate(predictions)
np.save(f'{path_to_results}/holdouts_iterative_2014-2023_predictions.npy', predictions)





