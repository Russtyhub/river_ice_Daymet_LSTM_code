#!/usr/bin/python3
# conda activate DL
# python3 04_calculating_results_tables.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import copy
from sklearn.metrics import mean_absolute_percentage_error

#################################################
# Script Parameters:
final_datasets_path = '/mnt/locutus/remotesensing/r62/river_ice_breakup/final_Daymet_datasets'
path_to_outputs = '/mnt/locutus/remotesensing/r62/river_ice_breakup/Results/DAYMET_Results/Output'
main_results_path = '/mnt/locutus/remotesensing/r62/river_ice_breakup/Results/DAYMET_Results/Output/Main'
figures_path = '/mnt/locutus/remotesensing/r62/river_ice_breakup/Images/Daymet_Results'
number_of_locations = 23
number_of_holdout_locations = 10
LOOKBACK_WINDOW = 457
#################################################

# Importing the data
main_dataset = pd.read_pickle(f'{final_datasets_path}/DAYMET_{number_of_locations}_LOCATIONS_PRE_LSTM.pkl')
main_mins_maxes = main_dataset['mins_maxes_df']
daymet_mins = np.array(main_mins_maxes['mins'])[0:15]
daymet_maxes = np.array(main_mins_maxes['maxes'])[0:15]

df = main_dataset['df']

# test_LH_Function = np.load(f'{main_results_path}/predictions_iteratively_main_{number_of_locations}_locations_old_version.npy')
test_LH_Function = np.load(f'{main_results_path}/predictions_iteratively_main_{number_of_locations}_locations.npy')

n = test_LH_Function.shape[0]
df = df.iloc[-n:, :]

train_y = main_dataset['train_y']
val_y = main_dataset['val_y']
test_y = main_dataset['test_y']
test_actuals = np.concatenate([train_y, val_y, test_y])[-n:]

sites = np.concatenate([main_dataset['train_location'], main_dataset['val_location'], main_dataset['test_location']])
# slick way to vectorize functions operating on np arrays
my_lambda = lambda x: x.replace('_', ' ')
remov_underscores = np.vectorize(my_lambda)
sites = remov_underscores(sites)
sites = sites[-n:]

val_preds = np.load(f'{path_to_outputs}/Main/val_predictions.npy')[1:]

# print()
# print('test_LH_Function', len(test_LH_Function))
# print('test_sites', len(sites))
# print('test_dates', len(df.index))
# print('test_df', len(df))
# print('test_actuals', len(test_actuals))
# print()

# Holdouts that look at the entire 1980 - 2023 time period
holdout_pkl_data = pd.read_pickle(f'{final_datasets_path}/DAYMET_{number_of_holdout_locations}_HOLDOUTS_PRE_LSTM.pkl')

holdouts_LH_Function = np.load(f'{path_to_outputs}/Holdouts/holdout_predictions_1980-2023.npy')[1:]
holdouts_actuals = holdout_pkl_data['actuals'][LOOKBACK_WINDOW:]
holdouts_sites = remov_underscores(holdout_pkl_data['locations'])[LOOKBACK_WINDOW:]
holdouts_dates = holdout_pkl_data['dates'][LOOKBACK_WINDOW:]
holdouts_df = holdout_pkl_data['df'][LOOKBACK_WINDOW:]
holdouts_mins = holdout_pkl_data['mins_maxes_df']['mins'][0:15]
holdouts_maxes = holdout_pkl_data['mins_maxes_df']['maxes'][0:15]

# Holdouts for 2014 - 2023 that use iterative model evaluation:
iter_holdouts_LH_Function = np.load(f'{path_to_outputs}/Holdouts/holdouts_iterative_2014-2023_predictions.npy')
mask_2014_2023 = holdouts_df.index.year.isin([2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])

# Creating plotting dfs and results tables

def create_plot_df_and_doy_df(normed_df,
                              dates, 
                              sites, 
                              mins, 
                              maxes, 
                              actuals, 
                              LH_Function, 
                              float_vars_to_include=['tmax(deg c)', 'tmin(deg c)'], 
                              un_normalize_df=True):
    DF = copy.copy(normed_df)

    if un_normalize_df:
        DF = (DF * (maxes - mins)) + mins

    # Creating plot data
    plot_data = DF[float_vars_to_include].copy()
    plot_data.loc[:, 'sites'] = sites
    plot_data.loc[:, 'actuals'] = actuals
    plot_data.loc[:, 'LH_Function'] = LH_Function
    plot_data.loc[:, 'dates'] = pd.to_datetime(dates)

    plot_data['Predicted_Breakup'] = 0
    plot_data.reset_index(inplace=True, drop=True)

    plot_data.loc[:, float_vars_to_include] = plot_data[float_vars_to_include].astype('float32')
    plot_data.loc[:, 'year'] = plot_data['dates'].dt.year
    plot_data.loc[:, 'DOY'] = plot_data['dates'].dt.dayofyear

    # Mark predicted breakups
    plot_data['Predicted_Breakup'] = plot_data.groupby(['sites', 'year'])['LH_Function'].transform(lambda x: (x == x.max()).astype(int))

    # Create DOY DataFrame
    doy_df_rows = []
    grouped = plot_data.groupby(['sites', 'year'])
    for (site, year), group in grouped:
        if group['actuals'].sum() == 1:
            predicted_doy = group.loc[group['Predicted_Breakup'] == 1, 'DOY'].values[0]
            actual_doy = group.loc[group['actuals'] == 1, 'DOY'].values[0]
            doy_df_rows.append({'sites': site, 'year': year, 'predicted_doy': predicted_doy, 'actual_doy': actual_doy})

    doy_df = pd.DataFrame(doy_df_rows, columns=['sites', 'year', 'predicted_doy', 'actual_doy'])
    doy_df = doy_df.astype({'year': 'int64', 'predicted_doy': 'int64', 'actual_doy': 'int64'})

    return plot_data, doy_df

plot_data_test, doy_df_test = create_plot_df_and_doy_df(df, 
		df.index, 
        sites, 
        daymet_mins, 
        daymet_maxes, 
        test_actuals, 
        test_LH_Function)
plot_data_test.to_pickle(f'{path_to_outputs}/Main/test_dataset_plotting.pkl')

plot_data_holdouts, doy_df_holdouts = create_plot_df_and_doy_df(holdouts_df,
		holdouts_dates,
        holdouts_sites,
        holdouts_mins, 
        holdouts_maxes, 
        holdouts_actuals, 
        holdouts_LH_Function)
plot_data_holdouts.to_pickle(f'{path_to_outputs}/Holdouts/holdout_dataset_plotting.pkl')

plot_data_iter_holdouts, doy_df_iter_holdouts = create_plot_df_and_doy_df(holdouts_df.loc[mask_2014_2023],
		holdouts_dates[mask_2014_2023],
        holdouts_sites[mask_2014_2023],
        holdouts_mins, 
        holdouts_maxes, 
        holdouts_actuals[mask_2014_2023], 
        iter_holdouts_LH_Function)
plot_data_iter_holdouts.to_pickle(f'{path_to_outputs}/Holdouts/holdout_iter_2024-2023_dataset_plotting.pkl')

# This is correct I checked it
def mape_function(x):
	return mean_absolute_percentage_error(x['actual_doy'], x['predicted_doy'])*100

doy_df_test['Residual'] = doy_df_test['actual_doy'] - doy_df_test['predicted_doy']
doy_df_test['Abs_Residual'] = doy_df_test['Residual'].abs()
overall_mape = mean_absolute_percentage_error(doy_df_test['actual_doy'],doy_df_test['predicted_doy'])
overall_mae = doy_df_test['Abs_Residual'].mean()
overall_std = doy_df_test['Abs_Residual'].std()

print()
print('Results from testing dataset')
print('OVERALL MAE:', overall_mae)
print('OVERALL STD:', overall_std)
print('OVERALL_MAPE:', 100*overall_mape)
print()

results_table = doy_df_test.groupby('sites')[['actual_doy','predicted_doy']].apply(mape_function).reset_index()
results_table.columns = ['Site', 'MAPE']
results_table['MAE'] = np.array(doy_df_test.groupby('sites').mean()['Abs_Residual'])
results_table['Min'] = np.array(doy_df_test.groupby('sites').min()['Abs_Residual'])
results_table['Max'] = np.array(doy_df_test.groupby('sites').max()['Abs_Residual'])
results_table['Std'] = np.array(doy_df_test.groupby('sites').std()['Abs_Residual'])
results_table = results_table[['Site', 'Min', 'Max', 'Std', 'MAE', 'MAPE']]
print(results_table.to_latex(index=False))
print()

doy_df_test.to_pickle(f'{main_results_path}/boxplot_data.pkl')

doy_df_holdouts['Residual'] = doy_df_holdouts['actual_doy'] - doy_df_holdouts['predicted_doy']
doy_df_holdouts['Abs_Residual'] = doy_df_holdouts['Residual'].abs()
overall_mape = mean_absolute_percentage_error(doy_df_holdouts['actual_doy'],doy_df_holdouts['predicted_doy'])
overall_mae = doy_df_holdouts['Abs_Residual'].mean()
overall_std = doy_df_holdouts['Abs_Residual'].std()

print('Results from holdouts dataset')
print('OVERALL_MAE:', overall_mae)
print('OVERALL_STD:', overall_std)
print('OVERALL_MAPE:', 100*overall_mape)
print()

results_table = doy_df_holdouts.groupby('sites')[['actual_doy','predicted_doy']].apply(mape_function).reset_index()
results_table.columns = ['Site', 'MAPE']
results_table['MAE'] = np.array(doy_df_holdouts.groupby('sites').mean()['Abs_Residual'])
results_table['Min'] = np.array(doy_df_holdouts.groupby('sites').min()['Abs_Residual'])
results_table['Max'] = np.array(doy_df_holdouts.groupby('sites').max()['Abs_Residual'])
results_table['Std'] = np.array(doy_df_holdouts.groupby('sites').std()['Abs_Residual'])
results_table = results_table[['Site', 'Min', 'Max', 'Std', 'MAE', 'MAPE']]
print(results_table.to_latex(index=False))
print()

###

doy_df_iter_holdouts['Residual'] = doy_df_iter_holdouts['actual_doy'] - doy_df_iter_holdouts['predicted_doy']
doy_df_iter_holdouts['Abs_Residual'] = doy_df_iter_holdouts['Residual'].abs()
overall_mape = mean_absolute_percentage_error(doy_df_iter_holdouts['actual_doy'],doy_df_iter_holdouts['predicted_doy'])
overall_mae = doy_df_iter_holdouts['Abs_Residual'].mean()
overall_std = doy_df_iter_holdouts['Abs_Residual'].std()

print('Results from holdouts using iterative evaluation (2014 - 2023)')
print('OVERALL_MAE:', overall_mae)
print('OVERALL_STD:', overall_std)
print('OVERALL_MAPE:', 100*overall_mape)
print()

results_table = doy_df_iter_holdouts.groupby('sites')[['actual_doy','predicted_doy']].apply(mape_function).reset_index()
results_table.columns = ['Site', 'MAPE']
results_table['MAE'] = np.array(doy_df_iter_holdouts.groupby('sites').mean()['Abs_Residual'])
results_table['Min'] = np.array(doy_df_iter_holdouts.groupby('sites').min()['Abs_Residual'])
results_table['Max'] = np.array(doy_df_iter_holdouts.groupby('sites').max()['Abs_Residual'])
results_table['Std'] = np.array(doy_df_iter_holdouts.groupby('sites').std()['Abs_Residual'])
results_table = results_table[['Site', 'Min', 'Max', 'Std', 'MAE', 'MAPE']]
print(results_table.to_latex(index=False))
print()

