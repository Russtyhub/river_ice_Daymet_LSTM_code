#!/usr/bin/python3

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib.colors import ListedColormap
import pickle
import subprocess
from datetime import datetime
from datetime import timedelta



def write_pickle(path, pickle_dictionary):
    with open(path, 'wb') as handle:
        pickle.dump(pickle_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(filepath):
	with open(filepath, 'rb') as handle:
		return pickle.load(handle)

class Color_Palettes:
    def __init__(self):
        self.color_blind_colors = ['#e34a33', '#fdbb84', '#2ca25f', '#99d8c9',
                                    '#9ebcda', '#8856a7', '#43a2ca', '#a8ddb5',
                                    '#2b8cbe', '#1c9099', '#dd1c77', '#c994c7', 
                                    '#636363', '#fa9fb5', '#c51b8a', '#fec44f',
                                    '#d95f0e', '#f03b20', '#756bb1', '#fc9272']
        
    def color_blind_friendly(self):
        return ListedColormap(self.color_blind_colors)

def find_list_mismatch(lis1, lis2):
    ''' Finds the elements of a list that are in one but not the other. 
    For Example:
    l1 = ['a', 'b', 'c']
    l2 = ['a', 'b', 'd']
    returns ['d', 'c']
    '''
    lis = (list(set(lis1).difference(lis2)))+(list(set(lis2).difference(lis1)))
    return list(set(lis))

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
    return np.array(predicted_breakup_dates)

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

def add_letters_to_subplots(axs,
							fontsize,
							left,
							up,
							start = 'A.)',
							one_letter = False,
							fontweight = None,
							with_periods = True):

	'''
	This function places letters in the top left corner of the 
	image for publication purposes. 
	
	If you do not use the ".)" part of the number (with_periods = False) 
	you need to make sure to change the start value to "A" or whatever 
	letter you wish to start with. This works great for subplots. For 
	example if you have an axis creatd from plt.subplots() you can simply
	use that axis in the first argument. left and up are for spacing how
	much to the left and up you want to adjust the letter.
	'''
	start = start.upper()
	if with_periods:
		letters = ['A.)', 'B.)', 'C.)', 'D.)', 'E.)', 'F.)',
				   'G.)', 'H.)','I.)', 'J.)', 'K.)', 'L.)', 'M.)',
				   'N.)', 'O.)', 'P.)', 'Q.)', 'R.)', 'S.)', 'T.)',
				   'U.)', 'V.)', 'X.)', 'Y.)', 'Z.)']
	else:
		letters = ['A', 'B', 'C', 'D', 'E', 'F',
				   'G', 'H','I', 'J', 'K', 'L', 'M',
				   'N', 'O', 'P', 'Q', 'R', 'S', 'T',
				   'U', 'V', 'X', 'Y', 'Z']

	letters = letters[letters.index(start):]

	if one_letter or len(letters) == 1:
		letter = letters[0]
		axs.text(-left, up,
					 letter,
					 transform=axs.transAxes,
					 fontsize=fontsize,
					 fontweight=fontweight,
					 verticalalignment='top')
	else:
		axs = np.array(axs)
		for idx, ax in enumerate(axs.flatten()):
			ax.text(-left, up,
					 letters[idx],
					 transform=ax.transAxes,
					 fontsize=fontsize,
					 fontweight=fontweight,
					 verticalalignment='top')
               
def pixel_extraction_tool_Daymet(start_date, end_date, latitude, longitude, VARS, output_path, as_csv=False, Return = False):

    '''
    Example:
    pixel_extraction_tool_Daymet(datetime(1980, 1, 1), '2022-12-31', 60.912732, -161.211508, 'all', '/mnt/locutus/remotesensing/r62/river_ice_breakup/Daymet_25_locations/test')

    This function will import daymet data as a pickle file (unless you change as_csv = True) given the point lat and lon coordinates provided 
    for a time series over the span of time provided by the start_date to end_date. 

    dates should be strings: 'year-month'day' while VARS should be a list of vars to use or 'all' indicating all
    variables of Daymet will be imported.

    Also be sure to include the name you want for the output file WITHOUT A FILE EXTENSION! in the output_path not just the directory & don't end in '/'
    this is an absolute path.

    '''

    if (type(VARS) == str) and (VARS.upper() == 'ALL'):
        VARS = 'dayl,prcp,srad,swe,tmax,tmin,vp'
        
    elif (type(VARS) == list) and (VARS != []):
        VARS = ",".join([str(VAR) for VAR in VARS])
        
    else:
        raise Exception("Something is wrong with the VARS argument")

    if type(start_date) == str:
        pass

    elif type(start_date) == datetime:
        
        year = start_date.year
        month = start_date.month
        day = start_date.day
        
        start_date = f'{year}-{month}-{day}'
        
    if type(end_date) == str:
        pass
        
    elif type(end_date) == datetime:
        
        year = end_date.year
        month = end_date.month
        day = end_date.day
        
        end_date = f'{year}-{month}-{day}'

    cmd = f"wget -O {output_path} --content-disposition 'https://daymet.ornl.gov/single-pixel/api/data?lat={latitude}&lon={longitude}&vars={VARS}&start={start_date}&end={end_date}'"
    runcmd(cmd)

    df = pd.read_csv(output_path, skiprows=list(np.arange(6)))
    os.remove(output_path)

    if as_csv:
        df.to_csv(output_path + '.csv')
    else:
        df.to_pickle(output_path + '.pkl')

    if Return:
        return df
    
class Pandas_Time_Converter():

    def __init__(self, df):
        self.df = df
            
        
    def convert_any_daily_timescale_to_datetime64(self, interval = 'daily', reference_date='start'):
        '''
        The index of the df must be the time axis being operated on.
        Each observation must represent a single day in the series
        reference_date = 'start' means the start date is taken
        from the first index value.
        '''
        if reference_date.upper() == 'START':
            ref_date = pd.to_datetime(str(self.df.index.to_numpy()[0]).split(' ')[0])
        else:
            ref_date = pd.to_datetime(reference_date)

        if interval.upper() == 'DAILY':
            datetime_dates = [ref_date + timedelta(days=i) for i in range(len(self.df))]
        elif interval.upper() == 'WEEKLY':
            datetime_dates = [ref_date + timedelta(days=i*7) for i in range(len(self.df))]
        elif interval.upper() == 'MONTHLY':
            datetime_dates = [ref_date + timedelta(days=i*30) for i in range(len(self.df))]

        return np.array(datetime_dates)

    def create_time_vars(self, date_column = None, create_year_var = True):
        
        '''
        The date column is where we assume the datetime info is. 
        If it is not specified (default) then the function assumes 
        the index is the datetime "column"
        '''
        
        DF = copy.deepcopy(self.df)
            
        if date_column == None:
            day_of_year = (DF.index.dayofyear - 1) * (360 / 364)
            radians = np.deg2rad(day_of_year)
            DF['COS_Radians'] = np.cos(radians)
            DF['SIN_Radians'] = np.sin(radians)
            
            if create_year_var:
                DF['Year'] = DF.index.year
            
        else:
            day_of_year = (DF[date_column].dt.dayofyear - 1) * (360 / 364)
            radians = np.deg2rad(day_of_year)
            DF['COS_Radians'] = np.cos(radians)
            DF['SIN_Radians'] = np.sin(radians)
            
            if create_year_var:
                DF['Year'] = DF[date_column].dt.year
            
        return DF

    def impute_df(self, impute_interval = 'daily'):
        if impute_interval.upper() == 'DAILY':
            return self.df.resample('D').ffill()
        
        elif impute_interval.upper() == 'WEEKLY':
            return self.df.resample('W').ffill()
        
        elif impute_interval.upper() == 'MONTHLY':
            return self.df.resample('M').ffill()
        
def split_train_val_test(DATA, train_prop, val_prop, how='sequential'):

    ''' Will seperate either numpy array or pandas dataframe on the 
    first axis into training, validation and testing in that order.

    if how == sequential:
    function assumes the data has already been sorted by your sequential variable
    (for example Date).    
    '''

    n = len(DATA)
    data = copy.copy(DATA)

    if type(data) == pd.core.series.Series or type(data) == pd.core.indexes.datetimes.DatetimeIndex or type(data) == list:
        data = np.array(data)
    else:
        pass

    if how.lower() == 'sequential':
        
        if type(data) == np.ndarray:

            train = data[0:int(n*train_prop)]
            val = data[int(n*train_prop):int(n*(train_prop + val_prop))]
            test = data[int(n*(train_prop + val_prop)):]

        elif type(data) == pd.core.frame.DataFrame:

            train = data.iloc[0:int(n*train_prop), :]
            val = data.iloc[int(n*train_prop):int(n*(train_prop + val_prop)), :]
            test = data.iloc[int(n*(train_prop + val_prop)):, :]
            
    elif how.lower() == 'random':
        
        if type(data) == np.ndarray:
            
            nums = np.random.choice(n, int(n*train_prop), replace = False)
            train = data[nums]
            mask = np.ones(n)
            mask[nums] = 0
            mask=mask.astype('bool')
            data = data[mask]

            nums = np.random.choice(len(data), int(len(data)*val_prop*(n/len(data))), replace = False)
            val = data[nums]
            mask = np.ones(len(data))
            mask[nums] = 0
            mask=mask.astype('bool')
            data = data[mask]

            if len(data) >= 1:
                test = data
            else:
                test = None
        
        elif type(data) == pd.core.frame.DataFrame:
            
            nums = np.random.choice(n, int(n*train_prop), replace = False)
            train = data.iloc[nums, :]
            data.drop(data.iloc[nums, :].index, inplace=True)

            nums = np.random.choice(len(data), int(len(data)*val_prop*(n/len(data))), replace = False)
            val = data.iloc[nums, :]
            data.drop(data.iloc[nums, :].index, inplace=True)

            if len(data) >= 1:
                test = data
            else:
                test = None
            
    return train, val, test

def normalize_df(df, col_list, convert, subtract_min = True):
    ''' Only works on float type columns unless you 
    set convert == True

    test_df = pd.DataFrame({'col1': [12, 13, 65, -1, 0], 'col2': [20, 3, 54, -3, 0],})
    test, MAXES, MINS = normalize_df(test_df, 'all', convert=True, subtract_min = True)

    TO CONVERT THE DATA BACK: (test*MAXES) + MINS
    '''

    DF = copy.copy(df)
    if type(col_list) == list:
        pass
    elif type(col_list) == str:
        if col_list.upper() == 'ALL':
            col_list = list(DF.columns)
        else:
            raise Exception("COL_LIST SHOULD BE EITHER 'ALL' OR A LIST OF YOUR COLUMNS")

    maxes=[]
    mins = []

    if subtract_min:
        for col in col_list:

            if (convert == True) and (np.nanmax(DF[col]) != 0) and (np.nanmax(DF[col]) != np.nanmin(DF[col])):
                DF[col] = DF[col].astype('float32')
                MIN = np.nanmin(DF[col])
                DF[col] = DF[col] - MIN
                MAX = np.nanmax(DF[col])
                DF[col] = DF[col]/MAX
                mins.append(MIN)
                maxes.append(MAX)

            elif (convert == False) and (DF[col].dtype == 'float') and (np.nanmax(DF[col]) != 0) and (np.nanmax(DF[col]) != np.nanmin(DF[col])):
                MIN = np.nanmin(DF[col])
                DF[col] = DF[col] - MIN
                MAX = np.nanmax(DF[col])
                DF[col] = DF[col]/MAX
                mins.append(MIN)
                maxes.append(MAX)

            else:
                raise Exception('ERROR SOMETHING IS WRONG HERE')

        return DF, maxes, mins

    else:
        for col in col_list:

            if (convert == True) and (np.nanmax(DF[col]) != 0) and (np.nanmax(DF[col]) != np.nanmin(DF[col])):
                DF[col] = DF[col].astype('float32')
                MAX = np.nanmax(DF[col])
                maxes.append(MAX)
                DF[col] = DF[col]/MAX

            elif (convert == False) and (DF[col].dtype == 'float') and (np.nanmax(DF[col]) != 0) and (np.nanmax(DF[col]) != np.nanmin(DF[col])):
                MAX = np.nanmax(DF[col])
                maxes.append(MAX)
                DF[col] = DF[col]/MAX
            else:
                raise Exception('ERROR SOMETHING IS WRONG HERE')

        return DF, maxes
    
def mask_df_to_x(df, mask, x):
    '''Where mask is a one-D np.array() with a length == df.shape[0]
    and x is a scalar value that will be the new "masked" value.'''

    DF = copy.copy(df)
    number_cols = DF.shape[1]
    DF[mask] = x*np.ones(number_cols)
    return DF

def runcmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)

def int_to_month_name(int_list, full_name=False):

    abreviated_names = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    full_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    if type(int_list[0]) == int:

        if full_name:
            output = [full_names[i] for i in int_list]
        else:
            output = [abreviated_names[i] for i in int_list]

    elif type(int_list[0]) == str:

        if full_name:
            full_name = [i.capitalize() for i in full_name]
            output = [full_names.index(i)+1 for i in int_list]
        else:
            abreviated_names = [i.capitalize() for i in abreviated_names]
            output = [abreviated_names.index(i)+1 for i in int_list]

    return output

def create_directory(directory_paths, PRINT = True):
    """
    Check if a directory exists, and create it if it doesn't.
    directory_paths must be a list of strings
    Parameters:
    - directory_paths (list): The path of the directory to check/create.
    """

    if isinstance(directory_paths, list):

        for DIR in directory_paths:
            if not os.path.exists(DIR):
                os.makedirs(DIR)
                if PRINT:
                    print(f"Directory '{DIR}' created.")
            else:
                if PRINT:
                    print(f"Directory '{DIR}' already exists.")
                else:
                    pass
    else:
        print('directory_paths should be a list type object')

def delete_everything_in_directory(dir_path, verbose=False):
    if not os.path.exists(dir_path):
        print(f"Directory {dir_path} does not exist.")
        return

    for root, dirs, files in os.walk(dir_path, topdown=False):
        # Delete files
        for name in files:
            file_path = os.path.join(root, name)
            try:
                os.remove(file_path)
                if verbose:
                    print(f"Deleted file: {file_path}", flush = True)
            except Exception as e:
                if verbose:
                    print(f"Error deleting file {file_path}: {e}", flush = True)
                else:
                    pass

        # Delete directories
        for name in dirs:
            dir_path = os.path.join(root, name)
            try:
                shutil.rmtree(dir_path)
                if verbose:
                    print(f"Deleted directory: {dir_path}", flush = True)
            except Exception as e:
                if verbose:
                    print(f"Error deleting directory {dir_path}: {e}", flush = True)
                else:
                    pass

    # Optionally, delete the root directory itself
    try:
        os.rmdir(dir_path)
        if verbose:
            print(f"Deleted root directory: {dir_path}", flush = True)
    except Exception as e:
        if verbose:
            print(f"Error deleting root directory {dir_path}: {e}", flush = True)
        else:
            pass

def df_to_LSTM(df, window_size):
    '''creates time series dataset'''
    if type(df) == pd.core.frame.DataFrame:
        df = df.to_numpy()
    X = [df[i:i + window_size] for i in range(len(df) - window_size)]
    return np.array(X)

def make_keras_tuner_trials_paths(number_of_trials, path):

    '''
    If you have more than 999 trials for HP tuning then 
    add to the code and question your decisions
    '''

    paths_to_create = []
    if number_of_trials >= 100:
        for trial in range(number_of_trials):
            trial_path = f'{path}/trial_{trial:03d}'
            paths_to_create.append(trial_path)

    elif (number_of_trials <= 99) and (number_of_trials >= 10):
        for trial in range(number_of_trials):
            trial_path = f'{path}/trial_{trial:02d}'
            paths_to_create.append(trial_path)

    elif number_of_trials < 10:
        for trial in range(number_of_trials):
            trial_path = f'{path}/trial_{trial:01d}'
            paths_to_create.append(trial_path)
            
    return paths_to_create

class Slurm_info():

    def __init__(self):
        self.slurms_node_list = os.environ.get('SLURM_JOB_NODELIST')
        self.nodes = self.get_nodes_list()
        
    def count_leading_zeros(self, s):
        count = 0
        for char in s:
            if char == '0':
                count += 1
            else:
                break
        return count

    def get_nodes_list(self):
        if not self.slurms_node_list or '[' not in self.slurms_node_list:
            return [self.slurms_node_list] if self.slurms_node_list else []
        else:
            string_split = self.slurms_node_list.split('[')
            machine = string_split[0]
            node_numbers = string_split[1].replace(']', '')
            node_numbers = node_numbers.split(',')
            nodes = []

            for n in node_numbers:
                if '-' in n:
                    vals = n.split('-')
                    pad_n_zeros = self.count_leading_zeros(vals[0])
                    padding = '0' * pad_n_zeros
                    MIN = int(vals[0])
                    MAX = int(vals[1])
                    nodes.extend([machine + padding + str(i) for i in range(MIN, MAX + 1)])
                else:
                    nodes.append(f'{machine}{n}')      

            return nodes

def check_trial_files(directory, remove_missing = False):
    missing_files = []
    for subdir in os.listdir(directory):
        if subdir.startswith("trial_"):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path):
                trial_file_path = os.path.join(subdir_path, 'trial.json')
                if not os.path.isfile(trial_file_path):
                    missing_files.append(subdir_path)

    if not missing_files:
        print("All 'trial' directories contain a 'trial.json' file", flush = True)
        return True
    else:
        print("The following 'trial' directories are missing 'trial.json' files:", flush = True)
        for missing in missing_files:
            print(missing)
        return False
    
def make_lists_equal_length(list1, list2):

    if len(list1) >= len(list2):
        newlist2 = list2*(math.ceil(len(list1)/len(list2)))
        lists_zipped = zip(list1, newlist2)
        
    if len(list2) > len(list1):
        newlist1 = list1*(math.ceil(len(list2)/len(list1)))
        lists_zipped = zip(newlist1, list2)

    return lists_zipped