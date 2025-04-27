#!/usr/bin/python3
# conda activate r62GEDI
# python3 05_Seasonal_Forecast_API_Request.py

# NOTES:
# This script is intended to download data from the ERA5 Seasonal Forecast dataset
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/seasonal-original-single-levels?tab=overview
# Best to just get the data in 6 hour intervals and process on your own
# it's unclear how aggregation works on their end.

import sys
sys.path.append('../')

import os
import pandas as pd
import cdsapi
import xarray as xr
# from urllib.request import urlopen
import warnings
import time
warnings.filterwarnings('ignore') 
import datetime

from resources import runcmd, read_pickle, write_pickle
directories = read_pickle('../.directories.pkl')

forecasts_path = os.path.join(directories['path_to_parent_directory'], 'forecasts/')
os.makedirs(forecasts_path, exist_ok=True)
directories['forecasts_path'] = forecasts_path
write_pickle('../.directories.pkl', directories)

c = cdsapi.Client()

def month_name_to_number(month_name, as_int = True):
	months = {'january': '01', 'jan' : '01', 'february': '02', 'feb': '02',
				'march': '03', 'mar': '03', 'april': '04', 'may': '05',
				'june': '06', 'july': '07', 'august': '08', 'aug': '08', 
				'september': '09', 'sept': '09', 'october': '10', 'oct': '10',
				'november': '11', 'nov': '11', 'december': '12', 'dec': '12'}

	if as_int:
		return int(months[month_name.lower()])
	else:
		return months[month_name.lower()]

def convert_number_of_days_to_hour_intervals(num_days, interval=6):

	if interval % 6 == 0:
		
		from math import ceil
		hours = num_days*24
		return [str(i) for i in range(interval, hours + interval, interval)]
		
	else:
		raise Exception('INTERVAL MUST BE DIVISIBLE BY 6')


##################################################################################################################

locations_path = os.path.join(directories['path_to_breakup_data'], 'summary_table_locations.pkl')
locations_csv = pd.read_pickle(locations_path)
locations_csv.reset_index(inplace = True)
locations_csv.rename(columns={'index' : 'Site'}, inplace = True)

dataset = 'seasonal-original-single-levels'

# Variables to import separated by type
static_vars = None
six_hour_vars = ['mean_sea_level_pressure', '2m_dewpoint_temperature']
twenty_four_hour_vars = ['surface_solar_radiation_downwards', 'maximum_2m_temperature_in_the_last_24_hours',
						 'minimum_2m_temperature_in_the_last_24_hours', 'snow_depth', 'snowfall',
						 'snow_density', 'total_column_water_vapour', 'total_precipitation',]
				
# lat_lon = (64.064750, -139.437167) # coordinates for Dawson Canada: HOWEVER I WON'T BE USING THIS NOW SINCE I HAVE A CSV OF LOCATIONS
# keep the above commented in case you want to do just a point location in the future the code is setup to work for that as well (switch 
# Loop_csv to False)
BUFFER = 0.1
OVERWRITE = True # to overwrite already downloaded files
# number_of_days = 151 # 02/01/2000 to 07/01/2000 = 151 days (151, 91, 61, 30)
keep_grib_index_file = False
FORMAT = 'grib' 
# will typically be grib, only change this if you know that the vars you want are available in netcdf

for start_month, number_of_days in zip(['March'], [122]):

	output_path = os.path.join(forecasts_path, start_month)
	start_month = month_name_to_number(start_month, as_int = False)

	if FORMAT.upper() == 'NETCDF':
		extension = '.nc'
	elif FORMAT.upper() == 'GRIB':
		extension = '.grib2' # I believe they tend to use .grib2 by default

	start = time.time()

	for REQUEST in ['six_hour_interval', 'twenty_four_hour_interval', 'static']:

		if REQUEST == 'six_hour_interval':
			VARIABLES = six_hour_vars
			lead_time_hours = convert_number_of_days_to_hour_intervals(number_of_days, 6)

		elif REQUEST == 'twenty_four_hour_interval':
			VARIABLES = twenty_four_hour_vars
			lead_time_hours = convert_number_of_days_to_hour_intervals(number_of_days, 24)

		elif REQUEST == 'static':
			VARIABLES = static_vars
			lead_time_hours = convert_number_of_days_to_hour_intervals(number_of_days, 24)

		if VARIABLES:

			print()
			print('FULFILLING REQUEST FOR SPECIFIED VARIABLES:', REQUEST.replace('_', ' ').upper())
			print()

			# API parameters 
			params = {
				'format': FORMAT.lower(),
				'variable': VARIABLES,
				'year':[2020, 2021, 2022, 2023],
				'month':[start_month], 
				'day': ['01'],
				'time': ['24:00'], # I think this makes the most sense
				'grid': [0.25, 0.25],
				'area': None,
				'system': '51',
				'leadtime_hour': lead_time_hours,
				'originating_centre': 'ecmwf',
				}


			for row_idx, row in locations_csv.iterrows():

				try:
					location = row['Site'].replace(' ', '_')
					lat_lon = (row['Latitude'], row['Longitude'])
					params['area'] = [lat_lon[0] + BUFFER, lat_lon[1] - BUFFER, lat_lon[0] - BUFFER, lat_lon[1] + BUFFER,]
					
					# retrieves the path to the file
					FILE_PATH = output_path + REQUEST + '_' + location + extension
					NEW_PATH = FILE_PATH.split('.')[0] + '.nc'

					if (os.path.exists(NEW_PATH)) and (OVERWRITE == False):
						continue

					elif (os.path.exists(NEW_PATH)) and (OVERWRITE == True):

						# Check when that file was downloaded:
						cutoff_date = datetime.datetime(2024, 10, 30)
						modification_time = os.path.getmtime(NEW_PATH)
						modification_time_readable = datetime.datetime.fromtimestamp(modification_time)

						if modification_time_readable < cutoff_date:
							pass
						else:
							print('PATH', NEW_PATH, 'IMPORTED RECENTLY')
							continue
					
					print('IMPORTING LOCATION:', location)
					fl = c.retrieve(dataset, params)
					fl.download(FILE_PATH)

					# open with xarray to convert to netcdf
					if FORMAT.upper() == 'GRIB':
						ds = xr.open_dataset(FILE_PATH, engine = 'cfgrib')
						ds.to_netcdf(NEW_PATH)
						runcmd(f'chmod 777 {NEW_PATH}')
						runcmd(f'rm {FILE_PATH}')

						if not keep_grib_index_file:
							index_file = [i for i in os.listdir(output_path) if i.endswith('.idx')][0]
							runcmd(f'rm {output_path}/{index_file}')

					print(REQUEST, 'variables downloaded for location:', location)
				
				except Exception as e:
					print('LOCATION:', location, 'FAILED TO IMPORT')

stop = time.time()
complete = round(((stop - start)/60), 2)
print('PROCESS COMPLETE TOOK:', complete, 'MINUTES')



