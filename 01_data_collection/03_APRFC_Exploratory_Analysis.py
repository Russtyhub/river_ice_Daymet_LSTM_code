#!/usr/bin/python3
# conda activate r62GEDI

# This script creates two plots: subplots_data_selection_{STYLE}.png and
# doy_over_time_{STYLE}.png where STYLE is either paper or poster. The 
# distinction between paper and poster is just for color and formatting 
# reasons. Figures are saved to wherever you want figures to go using
# the path_to_figures parameter below. subplots_data_selection_{STYLE}.png
# shows summary stats and distribution info for the entire APRFC database.
# It also shows my justification for how the number of locations were selected.
# doy_over_time_{STYLE}.png shows scatter plots of the Gregorian day of the 
# year that breakup occured for each river based on the data selected for
# given the previous exploratory step. A dataframe summarizing this info
# can be saved (save_df = True) to the same directory where you are keeping the breakup records
# imported from 01_Webscraper_Break-up_Data.py called summary_table_locations.pkl

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

with open("../.directories.pkl", "rb") as file:
    directories = pickle.load(file)

path_to_parent_directory = directories['path_to_parent_directory']
path_to_breakup_data = directories['path_to_breakup_data']
path_to_figures = os.path.join(path_to_parent_directory, 'Images')
os.makedirs(path_to_figures, exist_ok=True)

directories['path_to_figures'] = path_to_figures

with open("../.directories.pkl", "wb") as file:
    pickle.dump(directories, file)

#################################################
# Script Parameters:
STYLE = 'paper'
save_figures = True # if you want to save the figures created here
save_df = True # if you want to save the summary table
#################################################

sys.path.append('../')

# Useful functions
from resources import presentation_vs_poster, Color_Palettes, add_letters_to_subplots

# For figures
palettes = Color_Palettes()
col_blind_friendly_palette = palettes.color_blind_friendly()
color_blind_colors = palettes.color_blind_colors

# Import our breakup Data
break_up_data = pd.read_pickle(f'{path_to_breakup_data}/01_PROCESSED_BREAK_UP_DATA_WEBSCRAPED.pkl')

# Seeing how many total unique sites/rivers there are
number_of_unique_locations = []
for i in break_up_data.groupby(['River', 'Location']):
    number_of_unique_locations.append(i[0])
    
print('NUMBER OF UNIQUE LOCATIONS:', len(number_of_unique_locations))
print('NUMBER OF UNIQUE RIVERS/WATER BODIES:', np.unique(np.array(break_up_data['River'])).shape[0])

n_breakups_feb = len(break_up_data[break_up_data['Breakup Date'].dt.month == 2])

print('TIMES THE BREAKUP OCCURED IN FEBRUARY (EARLY):', n_breakups_feb)
print()

# Find the most complete locations from APRFC using simple optimization
minimum_number_of_dates = 20 # lowest number of breakup events I will consider for evaluation
start_year = 1980 # Start of Daymet there are 43 years from 1980 till 2023
end_year = break_up_data.Year.max()
optimal_number = 35 # 35 years with breakup events is determined below

summary_table = pd.DataFrame()
truncated_df = break_up_data[break_up_data.Year >= start_year]

for site, site_df in truncated_df.groupby(['Site']):
    min_year = site_df.Year.min()
    max_year = site_df.Year.max()
    years_available = site_df.Year
    years_test_against = np.arange(min_year, max_year+1)
    missing_years = list(np.setdiff1d(years_test_against, years_available))

    entry = {site: [len(years_available),
                       (min_year, max_year),
                        missing_years ]}
    entry = pd.DataFrame.from_dict(entry, orient='index')
    summary_table = pd.concat([summary_table, entry], ignore_index=False)

summary_table.columns = ['Number of Breakup Events', 'Year Range', 'Years Missing']
summary_table = summary_table[summary_table['Number of Breakup Events'] >= optimal_number]
summary_table.sort_index(inplace=True)

if save_df:
	summary_table.to_pickle(f'{path_to_breakup_data}/summary_table_locations.pkl')

location_counts = truncated_df['Site'].value_counts()

number_of_breakup_events, number_of_available_rivers = [], []

for threshold in np.arange(minimum_number_of_dates, end_year-start_year+1):
	number_of_locs_greater_than_thresh = (location_counts >= threshold).sum()
	sites_available = location_counts[location_counts >= threshold].index
	rivers_available = [(' ').join(i.split(' ')[-2:]) for i in sites_available]
	rivers_available = np.unique(rivers_available)
	number_of_available_rivers.append(len(rivers_available))
	number_of_breakup_events.append(number_of_locs_greater_than_thresh)

domain = np.arange(minimum_number_of_dates, end_year-start_year+1)

vis = presentation_vs_poster(STYLE, (16, 6))


fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])  # Top subplot is taller

# Create subplots separately
axs = []
axs.append(fig.add_subplot(gs[0, :]))  # First subplot spans both columns
axs.append(fig.add_subplot(gs[1, :]))  # Single bar plot spanning both columns

# Existing stem plot in axs[0] (unchanged)
axs[0].stem(domain, 
            number_of_breakup_events,
            markerfmt='D',    
            linefmt=vis['trendline_color'], 
            bottom=-1)

for x_i in range(len(domain)):
    number_of_rivers_avail = str(number_of_available_rivers[x_i])
    axs[0].annotate(number_of_rivers_avail, 
                    (int(domain[x_i]), int(number_of_breakup_events[x_i]) + 1),
                    ha='center', 
                    zorder=10,
                    fontsize='medium')

axs[0].set_xlabel(f'N: Minimum number of recorded breakup events \n to be considered from {start_year} - 2023', fontsize='large')
axs[0].set_ylabel('Number of Available Sites with \n at least N Breakup Dates', fontsize='large')
axs[0].axvline(optimal_number, linewidth=10, color='gold', linestyle='solid', zorder=1)
axs[0].grid(axis='y')
axs[0].set_ylim((0, 57))

# Data for bar plots
months = ['Feb', 'March', 'April', 'May', 'June']

counts_entire_record = [len(break_up_data.loc[break_up_data['Breakup Date'].dt.month == m]) for m in range(2, 7)]
counts_1980_to_2023 = [len(break_up_data.loc[(break_up_data['Breakup Date'].dt.month == m) & (break_up_data['Breakup Date'].dt.year >= 1980)]) for m in range(2, 7)]

# Define bar width and x positions for grouped bars
bar_width = 0.4
x = np.arange(len(months))  # X positions for bars

# Create a single bar plot with side-by-side bars
axs[1].bar(x - bar_width/2, counts_entire_record, color='cyan', edgecolor='black', width=bar_width, label='1896 - 2023')
axs[1].bar(x + bar_width/2, counts_1980_to_2023, color='coral', edgecolor='black', width=bar_width, label='1980 - 2023')

# Labeling and formatting
axs[1].set_xticks(x)
axs[1].set_xticklabels(months, fontsize='large')
axs[1].set_ylabel('Total Number \n of Recorded \n Breakup Events', fontsize='large')
axs[1].legend(fontsize='medium')
axs[1].grid(axis='y')

# Add subplot letters
add_letters_to_subplots(axs,
                        fontsize='x-large',
                        left=0.11,
                        up=1.2,
                        start='A.)',
                        one_letter=False,
                        fontweight=None,
                        with_periods=True)

plt.tight_layout()
if save_figures:
    plt.savefig(f'{path_to_figures}/subplots_data_selection_{STYLE}.png')




number_of_sites = number_of_breakup_events[np.where(np.arange(minimum_number_of_dates, end_year-start_year) == optimal_number)[0][0]]
print('NUMBER OF SITES WE CAN USE IS:', number_of_sites)
print('ASSUMING N =', optimal_number)

dist_plot_data = break_up_data[break_up_data.Site.isin(summary_table.index)]
dist_plot_data = dist_plot_data[dist_plot_data['Breakup Date'] >= f'01-01-{start_year}']
dist_plot_data['DayOfYear'] = dist_plot_data['Breakup Date'].dt.dayofyear

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
unique_rivers = ['Buckland River', 'Gakona River', 'Kobuk River',
                'Koyukuk River', 'Kuskokwim River', 'Susitna River',
                'Tanana River', 'Yukon River']
palette = sns.color_palette('Set3', n_colors=12)

print('RIVERS THAT FIT THE THRESHOLD:', unique_rivers)
print()

vis = presentation_vs_poster(STYLE, (16, 14))
# Create subplots
fig, ax = plt.subplots(nrows=len(unique_rivers), 
                       ncols=1, 
                       sharex=True, 
                       sharey=True, 
                       gridspec_kw={'hspace': 0}) # nrows=len(unique_rivers)

plt.subplots_adjust(left=0.1, right=None, bottom=None, top=None, wspace=None, hspace=0)

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
        ax[idx].scatter(site_data['Breakup Date'],
                      site_data['DayOfYear'],
                      label=(' ').join(site.split(' ')[:-2]),
                      linewidth=1.0, 
                      color = palettes.color_blind_colors[little_idx],
                      alpha = 0.7,
                      s = 50,
                      edgecolor = 'black')
    
    ax[idx].tick_params(axis='both', labelsize='large')

    shift = 1.05
    if river == 'Kuskokwim River':
        ax[idx].set_ylabel('Gregorian Day of \n the year breakup occurred', fontsize = 'x-large')
        ncol = 2       
    elif river == 'Kobuk River':
        ncol = 2        
    elif river == 'Yukon River':
        ncol = 2        
    else:
        ncol = 1

    if river == 'Gakona River':
        river = 'Copper River'
        
    ax[idx].grid(color = 'slategrey')
    legend = ax[idx].legend(title = river,
                            title_fontsize = 'x-large',
                            loc='upper left',
                             fontsize='large', 
                             bbox_to_anchor=(1, shift), 
                             facecolor = vis['FACECOLOR'],
                             ncol = ncol)
    legend.get_frame().set_edgecolor(vis['EDGE_color'])
    
    # for the sake of making it fit lets sacrifice a couple points
    ax[idx].set_xlim((pd.to_datetime('1980-01-01'), pd.to_datetime('2025-01-01')))
    ax[idx].set_ylim((95, 155))
    ax[idx].set_yticks([115, 130, 145])

ax[-1].set_xlabel('Time', fontsize = 'x-large')

plt.tight_layout()

if save_figures:
    plt.savefig(f'{path_to_figures}/doy_over_time_{STYLE}.png', dpi = 350)



