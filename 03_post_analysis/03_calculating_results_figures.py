#!/usr/bin/python3
# conda activate DL

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import sys
import copy
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

#################################################
# Script Parameters:
path_to_functions_dir = '/home/r62/repos/russ_repos/Functions/'
final_datasets_path = '/mnt/locutus/remotesensing/r62/river_ice_breakup/final_Daymet_datasets'
path_to_outputs = '/mnt/locutus/remotesensing/r62/river_ice_breakup/Results/DAYMET_Results/Output'
lstm_model_path = '/mnt/locutus/remotesensing/r62/river_ice_breakup/Results/DAYMET_Results/Models/DAYMET/DAYMET_BAYESIAN'
main_results_path = '/mnt/locutus/remotesensing/r62/river_ice_breakup/Results/DAYMET_Results/Output/Main'
figures_path = '/mnt/locutus/remotesensing/r62/river_ice_breakup/Images/Daymet_Results'
number_of_locations = 20
number_of_holdout_locations = 10
LOOKBACK_WINDOW = 457
#################################################

sys.path.append(path_to_functions_dir)
sys.path.append('../')
from STANDARD_FUNCTIONS import write_pickle, read_pickle
from TF_FUNCTIONS import df_to_LSTM, load_model
from RIVER_ICE_FUNCTIONS import presentation_vs_poster, date_difference, get_breakup_dates_by_yr

# Importing the data
main_dataset = pd.read_pickle(f'{final_datasets_path}/DAYMET_{number_of_locations}_LOCATIONS_PRE_LSTM.pkl')
main_mins_maxes = main_dataset['mins_maxes_df']
daymet_mins = np.array(main_mins_maxes['mins'])[0:15]
daymet_maxes = np.array(main_mins_maxes['maxes'])[0:15]

df = main_dataset['df']
print(main_dataset.keys())

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

print()
print('test_LH_Function', len(test_LH_Function))
print('test_sites', len(sites))
print('test_dates', len(df.index))
print('test_df', len(df))
print('test_actuals', len(test_actuals))
print()

# Holdouts
holdout_pkl_data = pd.read_pickle(f'{final_datasets_path}/DAYMET_{number_of_holdout_locations}_HOLDOUTS_PRE_LSTM.pkl')
holdouts_LH_Function = np.load(f'{path_to_outputs}/Holdouts/holdout_predictions_1980-2023.npy')[1:]

holdouts_actuals = holdout_pkl_data['actuals'][LOOKBACK_WINDOW:]
holdouts_sites = remov_underscores(holdout_pkl_data['locations'])[LOOKBACK_WINDOW:]
holdouts_dates = holdout_pkl_data['dates'][LOOKBACK_WINDOW:]
holdouts_df = holdout_pkl_data['df'][LOOKBACK_WINDOW:]
holdouts_mins = holdout_pkl_data['mins_maxes_df']['mins'][0:15]
holdouts_maxes = holdout_pkl_data['mins_maxes_df']['maxes'][0:15]
print()
print('holdouts_LH_Function', len(holdouts_LH_Function))
print('holdouts_sites', len(holdouts_sites))
print('holdouts_dates', len(holdouts_dates))
print('holdouts_df', len(holdouts_df))
print('holdouts_actuals', len(holdouts_actuals))
print()

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
    DF = normed_df.copy()

    if un_normalize_df:
        DF = (DF * (maxes - mins)) + mins

    # Creating plot data
    plot_data = DF[float_vars_to_include].copy()
    plot_data['sites'] = sites
    plot_data['actuals'] = actuals
    plot_data['LH_Function'] = LH_Function
    plot_data['dates'] = pd.to_datetime(dates)

    plot_data['Predicted_Breakup'] = 0
    plot_data.reset_index(inplace=True, drop=True)

    plot_data[float_vars_to_include] = plot_data[float_vars_to_include].astype('float32')
    plot_data['year'] = plot_data['dates'].dt.year
    plot_data['DOY'] = plot_data['dates'].dt.dayofyear

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

# fixing an error in the database I caught late (doesn't wreck our results by enough to mention)
# plot_data_test.loc[(plot_data_test.sites == 'Emmonak Yukon River') & (plot_data_test.year == 2024), 'Breakup Date'] = pd.to_datetime('2024-05-24')
# plot_data_test.loc[(plot_data_test.sites == 'Aniak Kuskokwim River') & (plot_data_test.year == 2023), 'Breakup Date'] = pd.to_datetime('2023-05-15')


plot_data_holdouts, doy_df_holdouts = create_plot_df_and_doy_df(holdouts_df,
		holdouts_dates,
        holdouts_sites,
        holdouts_mins, 
        holdouts_maxes, 
        holdouts_actuals, 
        holdouts_LH_Function)

# Fixing the mistake I caught late in APRFC (Year off from Breakup Date
# year)
mask = (plot_data_test.sites == 'Emmonak Yukon River') & (plot_data_test.actuals == 1) & (plot_data_test.dates == '2023-05-27')
# plot_data_test.loc[mask, ['dates', 'year']] = [pd.to_datetime('2024-05-24'), 2024]
# had to delete this one because 2024 is too late (as of now) for Daymet
plot_data_test = plot_data_test.loc[~mask]

print(plot_data_test[(plot_data_test.sites == 'Emmonak Yukon River') & (plot_data_test.actuals == 1)])
mask = (plot_data_test.sites == 'Aniak Kuskokwim River') & (plot_data_test.actuals == 1) & (plot_data_test.dates == '2022-05-15')
plot_data_test.loc[mask, ['dates', 'year']] = [pd.to_datetime('2023-05-15'), 2023]


# Create holdouts results table

# This is correct I checked it
def mape_function(x):
	return mean_absolute_percentage_error(x['actual_doy'], x['predicted_doy'])*100

doy_df_test['Residual'] = doy_df_test['actual_doy'] - doy_df_test['predicted_doy']
doy_df_test['Abs_Residual'] = doy_df_test['Residual'].abs()
overall_mape = mean_absolute_percentage_error(doy_df_test['actual_doy'],doy_df_test['predicted_doy'])
overall_mae = doy_df_test['Abs_Residual'].mean()
overall_std = doy_df_test['Abs_Residual'].std()

print('Results from testing dataset')
print('OVERALL MAE:', overall_mae)
print('OVERALL STD:', overall_std)
print('OVERALL_MAPE:', 100*overall_mape)

results_table = doy_df_test.groupby('sites')[['actual_doy','predicted_doy']].apply(mape_function).reset_index()
results_table.columns = ['Site', 'MAPE']
results_table['MAE'] = np.array(doy_df_test.groupby('sites').mean()['Abs_Residual'])
results_table['Min'] = np.array(doy_df_test.groupby('sites').min()['Abs_Residual'])
results_table['Max'] = np.array(doy_df_test.groupby('sites').max()['Abs_Residual'])
results_table['Std'] = np.array(doy_df_test.groupby('sites').std()['Abs_Residual'])
results_table = results_table[['Site', 'Min', 'Max', 'Std', 'MAE', 'MAPE']]
print(results_table)
print()

doy_df_holdouts['Residual'] = doy_df_holdouts['actual_doy'] - doy_df_holdouts['predicted_doy']
doy_df_holdouts['Abs_Residual'] = doy_df_holdouts['Residual'].abs()
overall_mape = mean_absolute_percentage_error(doy_df_holdouts['actual_doy'],doy_df_holdouts['predicted_doy'])
overall_mae = doy_df_holdouts['Abs_Residual'].mean()
overall_std = doy_df_holdouts['Abs_Residual'].std()

print('Results from holdouts dataset')
print('OVERALL_MAE:', overall_mae)
print('OVERALL_STD:', overall_std)
print('OVERALL_MAPE:', 100*overall_mape)

results_table = doy_df_holdouts.groupby('sites')[['actual_doy','predicted_doy']].apply(mape_function).reset_index()
results_table.columns = ['Site', 'MAPE']
results_table['MAE'] = np.array(doy_df_holdouts.groupby('sites').mean()['Abs_Residual'])
results_table['Min'] = np.array(doy_df_holdouts.groupby('sites').min()['Abs_Residual'])
results_table['Max'] = np.array(doy_df_holdouts.groupby('sites').max()['Abs_Residual'])
results_table['Std'] = np.array(doy_df_holdouts.groupby('sites').std()['Abs_Residual'])
results_table = results_table[['Site', 'Min', 'Max', 'Std', 'MAE', 'MAPE']]
print(results_table)
print()

###################################################################################
# Create boxplots
# for the testing dataset containing 20 locations
###################################################################################

box_plot_list = []
Years = []
for i in doy_df_test.groupby('year'):
    arr = np.array(i[1]['Residual'])
    box_plot_list.append(arr)
    Years.append(int(i[0]))
    
STYLE = 'paper'
vis = presentation_vs_poster(STYLE, (13.5, 7.2))

params = {'ytick.labelsize' : 18,
          'xtick.labelsize' : 18}

plt.rcParams.update(params)

# Horizontal lines
HLINES = [-14, -7, 0, 7, 14]

fig, ax = plt.subplots()

fig.patch.set_facecolor(vis['FACECOLOR'])
ax.set_facecolor(vis['FACECOLOR'])

for v in HLINES:
    if v == 0:
        ax.axhline(v, color=vis['EDGE_color'], alpha=0.8, zorder=0, linewidth = 2.5, ls = 'dotted')
    else:
        ax.axhline(v, color=vis['EDGE_color'], ls=(0, (5, 5)), alpha=0.8, zorder=0)
    
# The output is stored in 'violins', used to customize their appearence
violins = ax.violinplot(
    box_plot_list, 
    positions=Years,
    widths=0.2,
    bw_method="scott",
    showmeans=False, 
    showmedians=False,
    showextrema=False
)

# Customize violins (remove fill, customize line, etc.)
for pc in violins["bodies"]:
    pc.set_facecolor("none")
    pc.set_edgecolor('black')
    pc.set_linewidth(0.85)
    pc.set_alpha(1)
    
# Add boxplots ---------------------------------------------------
# Note that properties about the median and the box are passed
# as dictionaries.

medianprops = dict(
    linewidth=4, 
    color='turquoise',
    linestyle = 'solid',
    solid_capstyle="butt", 
    zorder = 5
)

if STYLE.lower() == 'paper':
    meanprops = dict(
        linewidth = 4,
        color = 'maroon',
        linestyle = 'solid',
        solid_capstyle="butt", 
        zorder = 5
    )

elif STYLE.lower() == 'poster':
    meanprops = dict(
        linewidth = 4,
        color = 'coral',
        linestyle = 'solid',
        solid_capstyle="butt", 
        zorder = 5
    )
    
flierprops = dict(marker='o', markerfacecolor=vis['FACECOLOR'], markersize=8, linestyle='none', markeredgecolor= vis['EDGE_color'])
capprops = dict(linewidth=2, color= vis['EDGE_color'])
whiskerprops = dict(linewidth=2, color=vis['EDGE_color'])  
boxprops = dict(linewidth=2, color=vis['EDGE_color'], zorder = 10)

ax.boxplot(
    box_plot_list,
    positions=Years, 
    showfliers = True, # show the outliers beyond the caps.
    showcaps = True,   # show the caps
    showbox=True,
    meanline = True,
    showmeans = True,
    meanprops = meanprops,
    medianprops = medianprops,
    boxprops = boxprops,
    whiskerprops=whiskerprops,
    flierprops=flierprops,
    capprops=capprops,
);

ax.grid(color = vis['FACECOLOR'])
ax.set_ylabel('Residuals (Days)', color= vis['EDGE_color'], fontsize = 'x-large')
ax.set_xlabel('Year of Breakup', color= vis['EDGE_color'], fontsize = 'x-large')
ax.set_xticks(Years)

plt.tight_layout()
plt.savefig(f'{figures_path}/boxplots_20_sites_{STYLE}.png');

###################################################################################
# Creating the subplots of the liklihood functions, temperature profiles
# and breakup prediction
###################################################################################

STYLE = 'paper'
vis = presentation_vs_poster(STYLE, (18, 11))

fig, axs = plt.subplots(nrows=5, ncols=4, sharex=True, sharey = 'all')
# plt.subplots_adjust(hspace=0.3, wspace=0.1)
plt.xticks(rotation=45)
props_1 = dict(boxstyle='round', facecolor='wheat', alpha=1.0, zorder = 20)
fig.patch.set_facecolor(vis['FACECOLOR'])

axs = axs.ravel()
for idx, (ax, DF) in enumerate(zip(axs, plot_data_test.groupby('sites'))):
    
    # a chunk of 2014 was still present as a flatline in the plot which was throwing off the function to find the local max
    # if DF[0] == 'Lake Creek Yentna River':
    #     data_by_site = DF[1]
    #     data_by_site = data_by_site[data_by_site.dates >= '2015-01-01']
    # else:
    #     data_by_site = DF[1]
        
    # data_by_site = DF[1]
    data_by_site = DF[1].sort_values('dates')
    true_break_up_dates = np.array(data_by_site['dates'][data_by_site['actuals'] == 1].dt.date)
    predicted_break_up_dates = get_breakup_dates_by_yr(data_by_site, liklihood_func_column='LH_Function')

    data_by_site.loc[data_by_site['dates'].isin(predicted_break_up_dates), 'Predicted_Breakup'] = 1
    
    ax.set_title(DF[0], fontsize = 'x-large')
    ax.tick_params(axis='y', colors=vis['second_axs_col'])
    
    if idx == 8:
        ax.set_ylabel('Temperature $(\u00b0C)$', color=vis['second_axs_col'], fontsize='x-large')
        
    ax.tick_params(axis='x', rotation=45)
    ax.fill_between(data_by_site['dates'],
                     np.array(data_by_site['tmin(deg c)']),
                     np.array(data_by_site['tmax(deg c)']),
                     color = vis['second_axs_col'],
                     label='Temperature',
                     zorder = 1)
    
    ax.set_facecolor(vis['FACECOLOR'])
            
    ax2 = ax.twinx()
       
    ax2.plot(data_by_site['dates'], data_by_site['LH_Function'], linewidth = 3, color = 'deepskyblue')
    ax2.scatter(predicted_break_up_dates, data_by_site['LH_Function'][data_by_site.dates.isin(predicted_break_up_dates)],
                     color = 'gold',
                     alpha=1.0,
                     zorder = 5, 
                     marker = 'd') # predicted
    
    ax2.vlines(true_break_up_dates,
                    ymin=0,
                    ymax=0.1,
                    color = 'purple',
                    linestyles = 'dashed',
                    linewidth = 2.5,
                    alpha=0.85,
                    zorder = 3) # actual
    
    ax2.set_xlabel('Time')
    
    if idx == 11:
        ax2.set_ylabel(r'$L(Breakup | X_{1...i})$', color = 'royalblue', fontsize = 'x-large')
    
    if idx in [3, 7, 11, 15, 19]:
        ax2.tick_params(axis='y', colors = 'royalblue')
        
    else:
        ax2.set_yticklabels([])

    differences, plot_dates = date_difference(data_by_site)

    for plot_date, diff in zip(plot_dates, differences):
        ax2.text(plot_date, 0.005, str(diff), ha='center', fontsize='large', bbox=props_1, color = 'black')
        
plt.tight_layout()
plt.savefig(f'{figures_path}/Daymet_{number_of_locations}_sites_forecast_{STYLE}.png', dpi = 450)





















