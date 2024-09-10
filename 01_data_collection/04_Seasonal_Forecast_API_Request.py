#!/usr/bin/python3
# conda activate r62GEDI
# python3 Seasonal_Forecast_API_Request.py

# NOTES:
# This script is intended to download data from the ERA5 Seasonal Forecast dataset
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/seasonal-original-single-levels?tab=overview
# Best to just get the data in 6 hour intervals and process on your own
# it's unclear how aggregation works on their end.

import sys
sys.path.append('/home/r62/repos/russ_repos/Functions')

import os
import pandas as pd
import cdsapi
import xarray as xr
# from urllib.request import urlopen
import warnings
import time
warnings.filterwarnings('ignore') 
import datetime

from ERA5_FUNCTIONS import convert_number_of_days_to_hour_intervals
from TIME import month_name_to_number
from STANDARD_FUNCTIONS import runcmd

c = cdsapi.Client()

##################################################################################################################

locations_csv = pd.read_pickle('/mnt/locutus/remotesensing/r62/river_ice_breakup/Breakup_Data/summary_table_locations.pkl')
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
# number_of_days = 151 # 02/01/2000 to 07/01/2000 = 151 days (151, 91, 61, 30)
keep_grib_index_file = False
FORMAT = 'grib' # will typically be grib only change this if you know that the vars you want are available in netcdf
Loop_csv = True
# overwrite = True # need this if the file exists with the same name from a previous run
# start_month = 'feb'
cutoff_date = datetime.datetime(2024, 9, 1)

# ['feb' 'mar', 'april', 'may', 'june']
# [151, 91, 61, 30]

for start_month, number_of_days in zip(['june'], [30]):
	start_month = start_month.lower()
	output_path = f'/mnt/locutus/remotesensing/r62/river_ice_breakup/ERA5/monthly_forecasts/differing_spans/{start_month}/' # Must have last slash
	start_month = month_name_to_number(start_month, as_int = False)
	##################################################################################################################

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

			# API parameters 
			params = {
				'format': FORMAT.lower(),
				'variable': VARIABLES,
				'year':[2020, 2021, 2022, 2023],
				'month':[start_month], #, '03', '04', '05', '06',],
				'day': ['01'],
				'time': ['24:00'], # I think this makes the most sense
				'grid': [0.25, 0.25],
				'area': None,
				'system': '51',
				'leadtime_hour': lead_time_hours,
				'originating_centre': 'ecmwf',
				}

			# This is where I can add multiple locations to loop through:
			if Loop_csv:

				for row_idx, row in locations_csv.iterrows(): 
					location = row['Site'].replace(' ', '_')
					lat_lon = (row['Latitude'], row['Longitude'])
					params['area'] = [lat_lon[0] + BUFFER, lat_lon[1] - BUFFER, lat_lon[0] - BUFFER, lat_lon[1] + BUFFER,]
					
					# retrieves the path to the file
					FILE_PATH = output_path + REQUEST + '_' + location + extension
					NEW_PATH = FILE_PATH.split('.')[0] + '.nc'

					if os.path.exists(NEW_PATH):

						# Check when that file was downloaded:
						modification_time = os.path.getmtime(NEW_PATH)
						modification_time_readable = datetime.datetime.fromtimestamp(modification_time)

						if modification_time_readable < cutoff_date:
							os.remove(NEW_PATH)
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

			else:

				# retrieves the path to the file
				params['area'] = [lat_lon[0] + BUFFER, lat_lon[1] - BUFFER, lat_lon[0] - BUFFER, lat_lon[1] + BUFFER,]
				FILE_PATH = output_path + REQUEST + extension
				NEW_PATH = FILE_PATH.split('.')[0] + '.nc'

				if os.path.exists(NEW_PATH) and overwrite:
					os.remove(NEW_PATH)

				fl = c.retrieve('seasonal-original-single-levels', params)
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

				print(REQUEST, 'variables downloaded for location')

stop = time.time()
complete = round(((stop - start)/60), 2)
print('PROCESS COMPLETE TOOK:', complete, 'MINUTES')



