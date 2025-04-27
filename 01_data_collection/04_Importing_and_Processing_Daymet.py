#!/usr/bin/python3
# conda activate r62GEDI

# This script imports the Daymet dataset using the pixel extraction tool
# maintained by ORNL DAAC. CSVs are imported for each location that are
# selected in the previous step, the data is extracted and added to a large
# pd dataframe which ultimately is made into a pickle file. The final dataset
# (saved to the name DAYMET_{number_of_locations}_LOCATIONS_PRE_LSTM.pkl)
# is normalized and masked to remove portions of the season for locations
# and years that do not contain a breakup event Mins and Maxes are also 
# saved to un-normalize the dataset later on. 

# A separate set of locations
# that are not used in the final model are also saved for additional 
# evaluation as DAYMET_{number_of_locations}_HOLDOUTS_PRE_LSTM.pkl.

import pandas as pd
import numpy as np
import os
import shutil
import sys
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Point
import warnings
import copy 

sys.path.append('./..')
from resources import read_pickle, write_pickle, pixel_extraction_tool_Daymet, Pandas_Time_Converter, split_train_val_test, normalize_df, mask_df_to_x
#################################################
# Script Parameters:
optimal_number = 35
PERC_TRAIN = 0.6
PERC_VAL = 0.2
build_daymet_and_holdout_dataset = True
save_daymet_csvs = True
#################################################

directories = read_pickle("../.directories.pkl") 

path_to_breakup_data = directories['path_to_breakup_data']
path_to_parent_directory = directories['path_to_parent_directory']

daymet_holding_binary_path = os.path.join(path_to_parent_directory, 'temp_daymet_data/')
os.makedirs(daymet_holding_binary_path, exist_ok=True)
	
arcade_dataset_path = os.path.join(path_to_parent_directory, 'ARCADE/')
os.makedirs(arcade_dataset_path, exist_ok=True)
directories['arcade_dataset_path'] = arcade_dataset_path

final_datasets_path = os.path.join(path_to_parent_directory, 'final_Daymet_datasets/')
os.makedirs(final_datasets_path, exist_ok=True)
directories['final_datasets_path'] = final_datasets_path

write_pickle("../.directories.pkl", directories) 

def move_row(df, row_idx_name):
	row_to_move = df.loc[[row_idx_name]]
	rest_of_df = df.drop(row_idx_name)
	return pd.concat([row_to_move, rest_of_df])


# The 5 Locations being used for Holdout. These locations will
# be used to evaluate the model on locations completely unseen 
# by the DL model
holdout_locations = ['Ambler Kobuk River', 'Ruby Yukon River',
				  'Nikolai Kuskokwim River', 'Bettles Koyukuk River',
				  'Eagle Yukon River', 'Alakanuk Yukon River', 'Anvik Yukon River',
				  'Kalskag Kuskokwim River', 'Sunshine Susitna River',
				  'Stony River Kuskokwim River']

# Import summary data from previous step
summary_table = pd.read_pickle(f'{path_to_breakup_data}/summary_table_locations.pkl')

# add holdout mask
summary_table['Holdouts'] = 0
summary_table.loc[summary_table.index.isin(holdout_locations), 'Holdouts'] = 1
summary_table['Holdouts'] = summary_table['Holdouts'].astype('bool')

# if you just want the name:
summary_table['Just Name'] = [(' ').join(i.split(' ')[0:-2]) for i in summary_table.index]

# Importing breakup data from previous step
break_up_data = pd.read_pickle(f'{path_to_breakup_data}/01_PROCESSED_BREAK_UP_DATA_WEBSCRAPED.pkl')

# Import ARCADE Dataset
ARCADE = gpd.read_file(f'{arcade_dataset_path}/ARCADE_v1_37_1km.shp', encoding='utf-8')
target_crs = CRS.from_epsg(4326)
ARCADE = ARCADE.to_crs(target_crs)
ARCADE_VARS = ['water_frac', 'twi_mean', 'perimeter', 'gravelius', 'cslope_mea']

# Importing location coordinates:
location_coordinates = pd.read_csv(f'{path_to_breakup_data}/breakup_locations_at_least_10_events_1980-2023.csv')
subset_lat_lon = location_coordinates[location_coordinates['Full Name'].isin(summary_table.index)]
subset_lat_lon.sort_values(by = 'Full Name', inplace = True)

summary_table['Latitude'] = np.array(subset_lat_lon['Latitude']).astype('float32')
summary_table['Longitude'] = np.array(subset_lat_lon['Longitude']).astype('float32')

# Updating the summary table with coordinates
summary_table.to_pickle(f'{path_to_breakup_data}/summary_table_locations.pkl')

# Combining dataframes into one large dataframe for model training
warnings.simplefilter("ignore")

dfs, holdouts, missing_arcade_locations = [], [], []

# I need Nikolia on top so I can use its data to fill
# the missing nearby locations
summary_table = move_row(summary_table, 'Nikolai Kuskokwim River')
print(summary_table.head())

for location_name, row in summary_table.iterrows():
    
	print('IMPORTING DAYMET DATA FOR LOCATION', location_name)
	location_name_underscored = location_name.replace(' ', '_')
	DF = pixel_extraction_tool_Daymet('1980-01-01',
			'2023-12-31',
			row.Latitude,
			row.Longitude,
			'all',
			f'{daymet_holding_binary_path}/{location_name_underscored}',
			as_csv=False,
			Return = True)

	if os.path.exists(f'{daymet_holding_binary_path}/{location_name_underscored}'):
		os.remove(f'{daymet_holding_binary_path}/{location_name_underscored}')

	# Changing the column names and sorting
	DF.columns = ['Year', 'yday', 'dayl(s)', 'prcp(mm/day)', 'srad(W/m^2)', 'swe(kg/m^2)','tmax(deg c)', 'tmin(deg c)', 'vp(Pa)']
	DF.sort_values(['Year', 'yday'], inplace = True)

	# Finding site coordinates to extract ARCADE data:
	LAT = summary_table[summary_table.index == location_name].Latitude[0]
	LON = summary_table[summary_table.index == location_name].Longitude[0]
	point_to_extract = Point(LON, LAT)
    
	# Sample the coordinates of the 5 static variables we are using from ARCADE dataset and add them to the dataframes
	ARCADE_data_to_add = np.array(ARCADE[ARCADE.geometry.contains(point_to_extract)][ARCADE_VARS], dtype = 'float32')

	# After visual inspection on QGIS, I concluded that the second 
	# polygon of the two overlapping is the correct one so we will always choose the second

	if ARCADE_data_to_add.shape == (1, 5) or (ARCADE_data_to_add.shape == (2, 5)):
		ARCADE_data_to_add = ARCADE_data_to_add[-1, :]
		DF[ARCADE_VARS] = np.tile(ARCADE_data_to_add, (len(DF), 1))

		if location_name_underscored == 'Nikolai_Kuskokwim_River':
			nikolai_arcade_data = copy.copy(ARCADE_data_to_add)

	elif ARCADE_data_to_add.shape == (0, 5):
		missing_arcade_locations.append(location_name_underscored)
		print('LOCATION ', location_name, 'ARCADE DATA UNAVAILABLE (OUTSIDE BOUNDARY)')
		print('IMPUTING DATA FROM NEARBY NIKOLAI')
		DF[ARCADE_VARS] = np.tile(nikolai_arcade_data, (len(DF), 1))

	# Adding a Breakup date column, creating a datetime index and removing 
	# duplicate dates. Arbitrarily keeping the first date if there is a duplicate
	# (should not be an issue having checked and editted the dataset manually)
	DF['Breakup Date'] = 0
	DF.index = pd.to_datetime(DF['Year'].astype(str) + DF['yday'].astype(str), format='%Y%j')
	DF = DF[~DF.index.duplicated(keep='first')]
    
	# Importing the breakup data for the given site
	df_breakup_data = break_up_data[break_up_data.Site == location_name]
    
	# Filling in the breakup dates with ones
	DF['Breakup Date'].loc[DF[DF.index.isin(df_breakup_data['Breakup Date'])].index] = 1
    
	# Adding the radians:
	converter = Pandas_Time_Converter(DF)
	DF = converter.create_time_vars()
    
	# Casting the variables:
	for var in ['dayl(s)', 'prcp(mm/day)', 'srad(W/m^2)', 'swe(kg/m^2)',
	'tmax(deg c)', 'tmin(deg c)', 'vp(Pa)', 'COS_Radians','SIN_Radians']:
		DF[var] = DF[var].astype('float32')

	for var in ['Year', 'yday', 'Breakup Date']:
		DF[var] = DF[var].astype('int16')
    
    # Applying the mask to months that have had breakup events (i.e. assumes that 
	# all breakup events occur within March, April, May or June)
	missing_years = row['Years Missing']
	MASK = (DF.index.month.isin([4, 5, 6, 7])) & (DF['Year'].isin(missing_years))
	DF['Mask'] = MASK
	DF['Location'] = location_name_underscored

    # combining all site locations to one list EXCEPT for the holdout locations
	if location_name in holdout_locations:
		holdouts.append(DF)
	else:
		dfs.append(DF)

if not save_daymet_csvs:
	shutil.rmtree(daymet_holding_binary_path)

# Combining dataframes into one large dataframe for model training
combined = pd.concat(dfs, axis=0)

# Sanity check AND Creating final dataset
dfs = []
for i in combined.groupby(['Year', 'Location']): # each grouping is 365 days
	if i[1].shape[0] != 365:
		print(i[0], 'HAS SHAPE:', i[1].shape)
	else:
		pass
    
	dfs.append(i[1])
    
print()
print('SANITY CHECK AND TRAINING/TUNING DATASET CREATION COMPLETE')

combined = pd.concat(dfs, axis=0)
combined.drop(['yday'], axis = 1, inplace = True)

number_of_locations = len(combined.Location.unique())
print('THERE ARE', number_of_locations, 'UNIQUE LOCATIONS IN THE MAIN DF')

# I need to save the min and maxes of the combined dataset
mins_maxes = pd.DataFrame()
mins_maxes['mins'] = combined.min()
mins_maxes['maxes'] = combined.max()

# Processing Holdout Locations
holdouts = pd.concat(holdouts, axis=0)

dfs = []
for i in holdouts.groupby(['Year', 'Location']): # each grouping is 365 days
	if i[1].shape[0] != 365:
		print(i[0], 'HAS SHAPE:', i[1].shape)
	else:
		pass
    
	dfs.append(i[1])
    
print('SANITY CHECK AND HOLDOUT DATASET CREATION COMPLETE')
print()
holdouts = pd.concat(dfs, axis=0)
holdouts.drop(['yday'], axis = 1, inplace = True)

print('LOCATIONS IN THIS LIST WERE IMPUTED USING DATA FROM NIKOLAI:')
print(missing_arcade_locations)

# Creating Final Pickle file for LSTM
mask = np.array(combined.pop('Mask'))
actuals = np.array(combined.pop('Breakup Date'))
locations = np.array(combined.pop('Location'))

# normalizing the locations here:
combined, _, _ = normalize_df(combined, 'all', convert=True, subtract_min = True)

dates = combined.index

train_df, val_df, test_df = split_train_val_test(combined, PERC_TRAIN, PERC_VAL, how='sequential')
train_y, val_y, test_y = split_train_val_test(actuals, PERC_TRAIN, PERC_VAL, how='sequential')
train_dates, val_dates, test_dates = split_train_val_test(dates, PERC_TRAIN, PERC_VAL, how='sequential')
train_location, val_location, test_location = split_train_val_test(locations, PERC_TRAIN, PERC_VAL, how='sequential')
train_mask, val_mask, test_mask = split_train_val_test(mask, PERC_TRAIN, PERC_VAL, how='sequential')

# implimenting the mask but not on test data
train_df = mask_df_to_x(train_df, train_mask, -1) # Looks correct based on check!
val_df = mask_df_to_x(val_df, val_mask, -1)

pkl_dict = {}
pkl_dict['df'] = combined
pkl_dict['mask'] = mask

pkl_dict['train_df'] = train_df
pkl_dict['val_df'] = val_df
pkl_dict['test_df'] = test_df

pkl_dict['train_y'] = train_y
pkl_dict['val_y'] = val_y
pkl_dict['test_y'] = test_y

pkl_dict['train_dates'] = train_dates
pkl_dict['val_dates'] = val_dates
pkl_dict['test_dates'] = test_dates

pkl_dict['train_location'] = train_location
pkl_dict['val_location'] = val_location
pkl_dict['test_location'] = test_location

pkl_dict['mins_maxes_df'] = mins_maxes

pkl_dict['NOTES'] = 'All data has been normlized to be between 0 and 1. \
The mins and maxes can be used to re-assign the data back to its original form. \
Input data is supplied by DayMet. Lookback = 457 = 365 + 92 (number of days in April + May + June + 1 \
which is masked for years that are missing breakup events)'

path_to_daymet_model_data = os.path.join(final_datasets_path, f'DAYMET_{number_of_locations}_LOCATIONS_PRE_LSTM.pkl')
write_pickle(path_to_daymet_model_data, pkl_dict)

# removing holdout mask and actuals 
mask = np.array(holdouts.pop('Mask')) # holdouts are not used for training so the mask is unnecessary here
actuals = np.array(holdouts.pop('Breakup Date'))
locations = np.array(holdouts.pop('Location'))
number_of_holdout_locations = len(np.unique(locations))

# If I normalize holdout using combined df's maxes and mins
# the mins are less than 0 and the maxes greater than one for some features in holdout
mins_maxes = pd.DataFrame()
mins_maxes['mins'] = holdouts.min()
mins_maxes['maxes'] = holdouts.max()

# Normalizing the holdouts seperately to not have problems in weights/biases:
holdouts_normed, _, _ = normalize_df(holdouts, 'all', convert=True, subtract_min = True)

pkl_dict = {}
pkl_dict['not_normed_df'] = holdouts
pkl_dict['df'] = holdouts_normed
pkl_dict['dates'] = holdouts.index
pkl_dict['locations'] = locations
pkl_dict['actuals'] = actuals
pkl_dict['mins_maxes_df'] = mins_maxes

path_to_holdout_data = os.path.join(final_datasets_path, f'DAYMET_{number_of_holdout_locations}_HOLDOUTS_PRE_LSTM.pkl')
write_pickle(path_to_holdout_data, pkl_dict)


