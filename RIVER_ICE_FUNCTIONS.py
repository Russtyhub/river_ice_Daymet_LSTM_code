#!/usr/bin/python3

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy

sys.path.append('/home/r62/repos/russ_repos/Functions')
from STANDARD_FUNCTIONS import find_list_mismatch


def get_missing_years(DF):
    missing_years = pd.DataFrame()
    SITE = []
    yr_miss = []
    yr_range = []
    sites = []
    number_of_events = []
    for site in DF.groupby(['Location', 'River']):

        x = site[1]
        sites.append(site[0])
        SITE.append(' '.join(site[0]))
        
        number_of_events.append(len(x))
        full_time_series = np.arange(x.Year.min(), x.Year.max())
        yr_miss.append(find_list_mismatch(full_time_series, np.array(x['Year'])))
        yr_range.append((x.Year.min(), x.Year.max()))
    
    missing_years['Number of Breakup Events'] = np.array(number_of_events)
    missing_years['Site'] = SITE
    missing_years['Year Range'] = yr_range
    missing_years['Years Missing'] = yr_miss
    missing_years.set_index('Site', inplace=True)
    
    return missing_years

def get_breakup_dates_by_yr(data_by_site, liklihood_func_column):
    d = copy.copy(data_by_site)
    year = d.set_index('dates').index.year
    d['year'] = year
    predicted_breakup_dates = []
    for yr in np.sort(np.unique(np.array(year))):
        data_for_year = d[d['year'] == yr]
        predicted_breakup_dates.append(data_for_year[data_for_year[liklihood_func_column] == data_for_year[liklihood_func_column].max()]['dates'].dt.date)
    return np.squeeze(np.array(predicted_breakup_dates))

def find_pred_vs_actual_mismatch(predicted_break_up_dates, true_break_up_dates):

    true_breakup_years = []
    pred_breakup_years = []

    for p in predicted_break_up_dates:
        pred_breakup_years.append(p.year)

    for t in true_break_up_dates:
        true_breakup_years.append(t.year)

    mismatch_years = find_list_mismatch(true_breakup_years, pred_breakup_years)
    
    if mismatch_years == []:
        mismatch_years = 'NONE'
    
    return mismatch_years

def date_difference(df_by_site):
    df_by_site.sort_values(by='year', inplace = True)
    differences, plot_dates = [], []
    for (key, data) in df_by_site.groupby('year'):
        predicted = data.loc[data.Predicted_Breakup == 1, 'dates']
        actual = data.loc[data.actuals == 1, 'dates']
        if predicted.shape != actual.shape:
            continue
        else:
            diff = predicted.dt.dayofyear.values - actual.dt.dayofyear.values
            differences.append(diff)
            plot_dates.append(pd.to_datetime(predicted))

    return np.concatenate(differences), pd.Series(plot_dates)

def get_distance(probs, true_labels, start_distance, increment):
    peaks = np.array([])
    
    if len(true_labels) > 1:
    
        while true_labels.sum() != len(peaks):
            peaks, _ = find_peaks(probs, distance=start_distance)
            start_distance += increment
            
    elif len(true_labels) == 1:
        
        while true_labels != len(peaks):
            peaks, _ = find_peaks(probs, distance=start_distance)
            start_distance += increment
            
    else:
        print('true_labels must be a list containing either \n an integer or the values themselves')
        
    return peaks


def presentation_vs_poster(style, fig_dimensions):
    
    '''
    fig_dimensions should be a list or tuple of (fig_width, fig_height)
    '''
    
    vis = {}
    
    base_font_size = 12
    fig_width = fig_dimensions[0]
    fig_height = fig_dimensions[1]
    aspect_ratio = fig_width / fig_height
    
    scale_factor_small = 0.2 * aspect_ratio + 0.6
    scale_factor_medium = 0.3 * aspect_ratio + 0.5
    scale_factor_large = 0.4 * aspect_ratio + 0.4
    scale_factor_x_large = 0.5 * aspect_ratio + 0.3
    scale_factor_xx_large = 0.6 * aspect_ratio + 0.2
    
    vis['small_font_size'] = base_font_size*scale_factor_small
    vis['medium_font_size'] = base_font_size*scale_factor_medium
    vis['large_font_size'] = base_font_size*scale_factor_large
    vis['x_large_font_size'] = base_font_size*scale_factor_x_large
    vis['xx_large_font_size'] = base_font_size*scale_factor_xx_large
    
    if style.lower() == 'paper':
        
        # For most of the plots
        vis['first_axs_col'] = 'slategray'
        vis['second_axs_col'] = 'maroon'
        vis['general_text_color'] = 'black'
        vis['EDGE_color'] = 'black'
        vis['FACECOLOR'] = "white"    
        vis['line_width'] = 2
        
        # selecting number of sites
        vis['trendline_color'] = 'blue'
        
        params = {'figure.dpi': 350,
                  'figure.facecolor' : vis['FACECOLOR'],
                  "ytick.color": vis['EDGE_color'],
                  "xtick.color": vis['EDGE_color'],
                  'xtick.labelsize': vis['large_font_size'],
                  'ytick.labelsize': vis['large_font_size'],
                  "axes.labelcolor": vis['EDGE_color'],
                  "axes.edgecolor": vis['EDGE_color'],
                  "axes.titlecolor": vis['EDGE_color'],
                  "font.size": vis['small_font_size'],
                  "font.family": "serif",
                  'axes.linewidth': vis['line_width'],
                  'legend.fontsize': vis['large_font_size'],
                  'axes.facecolor': vis['FACECOLOR'],
                  'axes.titlesize' : vis['x_large_font_size'],
                  'grid.linewidth': vis['line_width']/2,
                  'figure.figsize': [fig_width, fig_height],
                  'text.color': vis['EDGE_color'],}
    
    elif style.lower() == 'poster':
        
        # For most of the plots
        vis['first_axs_col'] = 'beige'
        vis['second_axs_col'] = 'cyan'
        vis['general_text_color'] = 'white'
        vis['EDGE_color'] = 'white'
        vis['FACECOLOR'] = "#001543" # Dark Blue
        vis['line_width'] = 3
        
        # selecting number of sites
        vis['trendline_color'] = 'cyan'
        
        params = {'figure.dpi': 350,
                  'figure.facecolor' : vis['FACECOLOR'],
                  "ytick.color": vis['EDGE_color'],
                  "xtick.color": vis['EDGE_color'],
                  'xtick.labelsize': vis['large_font_size'],
                  'ytick.labelsize': vis['large_font_size'],
                  "axes.labelcolor": vis['EDGE_color'],
                  "axes.edgecolor": vis['EDGE_color'],
                  "axes.titlecolor": vis['EDGE_color'],
                  "font.size": vis['small_font_size'],
                  "font.family": "serif",
                  'axes.linewidth': vis['line_width'],
                  'legend.fontsize': vis['large_font_size'],
                  'axes.facecolor': vis['FACECOLOR'],
                  'axes.titlesize' : vis['x_large_font_size'],
                  'grid.linewidth': vis['line_width']/2,
                  'figure.figsize': [fig_width, fig_height],
                  'text.color': vis['EDGE_color'],}
    
    plt.rcParams.update(params)
    
    return vis