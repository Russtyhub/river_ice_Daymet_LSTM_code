#!/usr/bin/python3
# conda activate r62GEDI

# This script imports the Daymet dataset using the pixel extraction tool
# maintained by ORNL DAAC. CSVs are imported for each location that was
# selected in the previous step, the data is extracted and added to a large
# pd dataframe which ultimately is made into a pickle. The final dataset
# is normalized and masked to remove portions of the season for locations
# and years that do not contain a breakup event Mins and Maxes are also 
# saved to un-normalize the dataset later on. 
# saved to the name DAYMET_{number_of_locations}_LOCATIONS_PRE_LSTM.pkl 

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

#################################################
# Script Parameters:
path_to_breakup_records = '/mnt/locutus/remotesensing/r62/river_ice_breakup/Breakup_Data'
path_to_functions_dir = '/home/r62/repos/russ_repos/Functions/'
daymet_holding_binary_path = '/mnt/locutus/remotesensing/r62/river_ice_breakup/temp_daymet_data'
arcade_dataset_path = '/mnt/locutus/remotesensing/r62/river_ice_breakup/ARCADE'
final_datasets_path = '/mnt/locutus/remotesensing/r62/river_ice_breakup/final_Daymet_datasets'
optimal_number = 35
PERC_TRAIN = 0.6
PERC_VAL = 0.2
build_daymet_and_holdout_dataset = True
save_daymet_csvs = True
#################################################

if not os.path.exists(daymet_holding_binary_path):
    os.mkdir(daymet_holding_binary_path)

sys.path.append(path_to_functions_dir)
sys.path.append('../')

from REMOTE_SENSING_FUNCTIONS import pixel_extraction_tool_Daymet
from TIME import Pandas_Time_Converter
from DATA_ANALYSIS_FUNCTIONS import split_train_val_test, normalize_df
from STANDARD_FUNCTIONS import write_pickle, mask_df_to_x

# The 5 Locations being used for Holdout. These locations will
# be used to evaluate the model on locations completely unseen 
# by the DL model
holdout_locations = ['Ambler Kobuk River', 'Kaltag Yukon River',
                     'Nikolai Kuskokwim River', 'Bettles Koyukuk River',
                     'Eagle Yukon River', 'Alakanuk Yukon River', 'Anvik Yukon River',
                     'Kalskag Kuskokwim River', 'Sleetmute Kuskokwim River',
                     'Stony River Kuskokwim River']

# Import summary data from previous step
summary_table = pd.read_pickle(f'{path_to_breakup_records}/summary_table_locations.pkl')

# Importing breakup data from previous step
break_up_data = pd.read_pickle(f'{path_to_breakup_records}/01_PROCESSED_BREAK_UP_DATA_WEBSCRAPED.pkl')

# Import ARCADE Dataset
ARCADE = gpd.read_file(f'{arcade_dataset_path}/ARCADE_v1_37_1km.shp', encoding='utf-8')
target_crs = CRS.from_epsg(4326)
ARCADE = ARCADE.to_crs(target_crs)
ARCADE_VARS = ['water_frac', 'twi_mean', 'perimeter', 'gravelius', 'cslope_mea']

# Importing location coordinates:
location_coordinates = pd.read_csv(f'{path_to_breakup_records}/breakup_locations_at_least_10_events_1980-2023.csv')
subset_lat_lon = location_coordinates[location_coordinates['Full Name'].isin(summary_table.index)]
summary_table['Latitude'] = np.array(subset_lat_lon['Latitude']).astype('float32')
summary_table['Longitude'] = np.array(subset_lat_lon['Longitude']).astype('float32')

# Updating the summary table with coordinates
summary_table.to_pickle(f'{path_to_breakup_records}/summary_table_locations.pkl')

# Combining dataframes into one large dataframe for model training
warnings.simplefilter("ignore")

# if build_daymet_and_holdout_dataset:
#     summary_table = summary_table
# else:
#     summary_table = all_locations_at_least_10_breakups_1980_to_2023

dfs = []
holdouts = []

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

    # After visual inspection on QGIS using the code below I concluded that the second 
    # polygon of the two overlapping is the correct one so we will always choose the second
    
#     if ARCADE_data_to_add.shape == (2, 5):
#         print('LOCATION:', location_name, 'OVERLAPS TWO POLYGONS NEED TO CHOOSE MANUALLY')
#         print(ARCADE[ARCADE.geometry.contains(point_to_extract)][ARCADE_VARS])
#         print()
#         continue
        
#     else:
#         DF[ARCADE_VARS] = np.tile(ARCADE_data_to_add, (len(DF), 1))
    
    if ARCADE_data_to_add.shape == (2, 5):
        ARCADE_data_to_add = ARCADE_data_to_add[-1, :]
    elif ARCADE_data_to_add.shape == (0, 5):
        print('SKIPPING ', location_name, 'OUTSIDE OF ARCADE SHAPEFILE')
        continue
    
	# I lose 2 locations as a result of them not overlapping the ARCADE shapefile
    DF[ARCADE_VARS] = np.tile(ARCADE_data_to_add, (len(DF), 1))
    
    # Adding a Breakup date column, creating a datetime index and removing 
	# duplicate dates. Arbitrarily keeping the first date if there is a duplicate
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
                'tmax(deg c)', 'tmin(deg c)', 'vp(Pa)', 'COS_Radians',
                'SIN_Radians']:
        DF[var] = DF[var].astype('float32')
        
    for var in ['Year', 'yday', 'Breakup Date']:
        DF[var] = DF[var].astype('int16')
    
    # Applying the mask to months that have had breakup events (i.e. assumes that 
	# all breakup events occur within March, April and May)
    missing_years = row['Years Missing']
    MASK = (DF.index.month.isin([4, 5, 6])) & (DF['Year'].isin(missing_years))
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

write_pickle(f'{final_datasets_path}/DAYMET_{number_of_locations}_LOCATIONS_PRE_LSTM.pkl', pkl_dict)

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

write_pickle(f'{final_datasets_path}/DAYMET_{number_of_holdout_locations}_HOLDOUTS_PRE_LSTM.pkl', pkl_dict)


