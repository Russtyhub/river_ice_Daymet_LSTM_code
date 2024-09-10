#!/usr/bin/python3
# conda activate r62GEDI

# This script creates two plots: subplots_data_selection_{STYLE}.png and
# doy_over_time_{STYLE}.png where STYLE is either paper or poster. The 
# distinction between paper and poster is just for color and formatting 
# reasons. Figures are saved to wherever you want figures to go using
# the path_to_figures parameter below. subplots_data_selection_{STYLE}.png
# shows summary stats and distribution info for the entire APRFC database.
# It also shows my justification for how the number of locations was selected.
# doy_over_time_{STYLE}.png shows scatter plots of the Gregorian day of the 
# year that breakup occured for each river based on the data selected for
# given the previous exploratory step. A dataframe summarizing this info
# is saved to the same directory where you are keeping the breakup records
# imported from 01_Webscraper_Break-up_Data.py called summary_table_locations.pkl

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

#################################################
# Script Parameters:
path_to_breakup_records = '/mnt/locutus/remotesensing/r62/river_ice_breakup/Breakup_Data'
path_to_figures = '/mnt/locutus/remotesensing/r62/river_ice_breakup/Images/Daymet_Results'
path_to_functions_dir = '/home/r62/repos/russ_repos/Functions/'
STYLE = 'paper'
save_figures = True # if you want to save the figures created here
save_df = True # if you want to save the summary table
#################################################

sys.path.append(path_to_functions_dir)
sys.path.append('../')

# Useful functions
from STANDARD_FUNCTIONS import find_list_mismatch
from VISUALIZATION_FUNCTIONS import Color_Palettes
from RIVER_ICE_FUNCTIONS import presentation_vs_poster, get_missing_years

# For figures
palettes = Color_Palettes()
col_blind_friendly_palette = palettes.color_blind_friendly()
color_blind_colors = palettes.color_blind_colors

# Import our breakup Data
break_up_data = pd.read_pickle(f'{path_to_breakup_records}/01_PROCESSED_BREAK_UP_DATA_WEBSCRAPED.pkl')

# Seeing how many total unique sites/rivers there are
number_of_unique_locations = []
for i in break_up_data.groupby(['River', 'Location']):
    number_of_unique_locations.append(i[0])
    
print('NUMBER OF UNIQUE LOCATIONS:', len(number_of_unique_locations))
print('NUMBER OF UNIQUE RIVERS/WATER BODIES:', np.unique(np.array(break_up_data['River'])).shape[0])

print()
print('TIMES THE BREAKUP OCCURED IN FEBRUARY (EARLY):')
print(len(break_up_data[break_up_data['Breakup Date'].dt.month == 2]))

# Find the most complete locations from APRFC using simple optimization
minimum_number_of_dates = 20 # lowest number of breakup events I will consider for evaluation
start_year = 1980 # Start of Daymet there are 43 years from 1980 till 2023
end_year = break_up_data.Year.max()
optimal_number = 35 # 35 years with breakup events is determined below

summary_table = pd.DataFrame()
truncated_df = break_up_data[break_up_data.Year >= start_year]

for site in truncated_df.groupby(['Site']):
    min_year = site[1].Year.min()
    max_year = site[1].Year.max()
    years_available = site[1].Year
    years_test_against = np.arange(min_year, max_year+1)
    missing_years = list(np.setdiff1d(years_test_against, years_available))

    entry = {site[0]: [len(years_available),
                       (min_year, max_year),
                        missing_years ]}
    entry = pd.DataFrame.from_dict(entry, orient='index')
    summary_table = pd.concat([summary_table, entry], ignore_index=False)

summary_table.columns = ['Number of Breakup Events',	'Year Range', 'Years Missing']
summary_table = summary_table[summary_table['Number of Breakup Events'] > optimal_number]
summary_table.sort_index(inplace=True)

location_counts = truncated_df['Site'].value_counts()
number_of_breakup_events = []
for threshold in np.arange(minimum_number_of_dates, end_year-start_year+1):
	number_of_locs_greater_than_thresh = (location_counts >= threshold).sum()
	number_of_breakup_events.append(number_of_locs_greater_than_thresh)
	
if save_df:
    summary_table.to_pickle(f'{path_to_breakup_records}/summary_table_locations.pkl')

vis = presentation_vs_poster(STYLE, (20, 8))

fig, axs = plt.subplots(nrows=1, ncols=2, sharey=False)
plt.subplots_adjust(hspace=0.25)

axs[0].plot(np.arange(minimum_number_of_dates, end_year-start_year+1), number_of_breakup_events, linewidth = vis['line_width'], color = vis['trendline_color']);
axs[0].set_xlabel(f'N: Minimum number of recorded breakup events \n to be considered from {start_year} - Present', fontsize = 'x-large')
axs[0].set_ylabel(' Number of Available Sites with \n at least N Breakup Dates', fontsize = 'x-large')
axs[0].axvline(optimal_number, linewidth = vis['line_width'], color = 'magenta', linestyle = 'dashed')
axs[0].set_title('Number of Sites with Available Data vs Minimum \n Number of Breakup Events per site (N) \n', fontsize = 'xx-large')
axs[0].grid()

axs[1].bar('February', len(break_up_data[break_up_data['Breakup Date'].dt.month == 2]), color='coral', edgecolor = 'black')
axs[1].bar('March', len(break_up_data[break_up_data['Breakup Date'].dt.month == 3]), color='coral', edgecolor = 'black')
axs[1].bar('April', len(break_up_data[break_up_data['Breakup Date'].dt.month == 4]), color='coral', edgecolor = 'black')
axs[1].bar('May', len(break_up_data[break_up_data['Breakup Date'].dt.month == 5]), color='coral', edgecolor = 'black')
axs[1].bar('June', len(break_up_data[break_up_data['Breakup Date'].dt.month == 6]), color='coral', edgecolor = 'black')
axs[1].set_xlabel('Months Containing Breakup Events', fontsize = 'x-large')
axs[1].set_ylabel('Total Number of Recorded Breakup Events', fontsize = 'x-large')
axs[1].set_title('Number of Breakup Events Recorded by Month \n', fontsize = 'xx-large');

plt.tight_layout()
if save_figures:
    plt.savefig(f'{path_to_figures}/subplots_data_selection_{STYLE}.png');

number_of_sites = number_of_breakup_events[np.where(np.arange(minimum_number_of_dates, end_year-start_year) == optimal_number)[0][0]]
print('NUMBER OF SITES WE CAN USE IS:', number_of_sites)
print('ASSUMING N =', optimal_number)

dist_plot_data = break_up_data[break_up_data.Site.isin(summary_table.index)]
dist_plot_data = dist_plot_data[dist_plot_data['Breakup Date'] >= f'01-01-{start_year}']
dist_plot_data['DayOfYear'] = dist_plot_data['Breakup Date'].dt.dayofyear

# def return_site_info(site_data):
#     missing_years = np.setdiff1d(np.arange(start_year, end_year), np.array(site_data.Year))
#     for missing_yr in missing_years:
#         new_row = {'River' : river, 'Location' : 'Blank',
#                    'Year' : missing_yr, 'Breakup Date' : pd.to_datetime(f'{missing_yr}-01-01'), 
#                    'Site' : site, 'DayOfYear' : np.nan}

#         site_data = site_data.append(new_row, ignore_index=True)

#     return site_data.sort_values(by=['Year'])

def return_site_info(site_data):
    missing_years = np.setdiff1d(np.arange(start_year, end_year), np.array(site_data.Year))
    new_rows = []

    for missing_yr in missing_years:
        new_row = pd.DataFrame({
            'River': [river],
            'Location': ['Blank'],
            'Year': [missing_yr],
            'Breakup Date': [pd.to_datetime(f'{missing_yr}-01-01')],
            'Site': [site],
            'DayOfYear': [np.nan]
        })
        new_rows.append(new_row)

    if new_rows:
        site_data = pd.concat([site_data] + new_rows, ignore_index=True)

    return site_data.sort_values(by=['Year'])


presentation_vs_poster(STYLE, (12, 14))

# Get unique rivers in the dataset
unique_rivers = dist_plot_data['River'].unique()
palette = sns.color_palette('Set3', n_colors=12)

unique_rivers = list(unique_rivers)
unique_rivers.remove("Susitna River")
unique_rivers.remove("Gakona River")

# Create subplots
fig, axs = plt.subplots(nrows=len(unique_rivers), ncols=1, sharex=True, sharey=True)
# plt.subplots_adjust(left=0.1, right=None, bottom=None, top=None, wspace=None, hspace=0.1)

# Iterate through rivers
for idx, river in enumerate(unique_rivers):
    river_data = dist_plot_data[dist_plot_data['River'] == river]
    
    # Get unique sites for the current river
    unique_sites = river_data['Site'].unique()
    
    if river == 'Koyukuk River':
        alpha = 0.5
    else:
        alpha = 0.7

    # Plot each site on the current subplot
    for little_idx, site in enumerate(unique_sites):
        site_data = river_data[river_data['Site'] == site]
        site_data = return_site_info(site_data)
        axs[idx].scatter(site_data['Breakup Date'],
                      site_data['DayOfYear'],
                      label=(' ').join(site.split(' ')[:-2]),
                      linewidth=1.0, 
                      color = palettes.color_blind_colors[little_idx],
                      alpha = 0.7)
    
    axs[idx].set_title(river, fontsize='x-large')
    
    if river == 'Kobuk River':
        axs[idx].set_ylabel('Day of the year breakup occurred', fontsize = 'x-large')
        ncol = 2
    elif river == 'Yukon River':
        ncol = 2
    else:
        ncol = 1
    
    axs[idx].grid(color = 'slategrey')
    legend = axs[idx].legend(loc='upper left',
                             fontsize='large', 
                             bbox_to_anchor=(1, 1.065), 
                             facecolor = vis['FACECOLOR'],
                             ncol = ncol)
    legend.get_frame().set_edgecolor(vis['EDGE_color'])
    
axs[-1].set_xlabel('Time', fontsize = 'x-large')
plt.tight_layout()

if save_figures:
    plt.savefig(f'{path_to_figures}/doy_over_time_{STYLE}.png', dpi = 350)



