#!/usr/bin/python3
# conda activate r62GEDI


# this code pulls breakup dates from the Alaska-Pacific River Forecast Center Database
# The code runs slowly because the driver needs a chance to sleep (time.sleep(3)).
# Data is collected and saved as a CSV at a path of your choice with the title:
# 01_BREAK_UP_DATA_WEBSCRAPED.csv
# NOTE: there are some additional manual edits at the end of the code
# for errors in the import that I had caught

import os
import sys

sys.path.append('../')
from resources import write_pickle

import pandas as pd
from selenium import webdriver
#from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import time

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

start_time = time.time()

path_to_parent_directory = input('Enter the absolute path to where you want your data imported to:')
if path_to_parent_directory.endswith('/'):
	path_to_parent_directory = path_to_parent_directory[:-1]

directories = {}
directories['path_to_parent_directory'] = path_to_parent_directory

path_to_breakup_data = os.path.join(path_to_parent_directory, 'Breakup_Data')
directories['path_to_breakup_data'] = path_to_breakup_data

write_pickle("../.directories.pkl", directories)

holding_dir = f'{path_to_breakup_data}/holding_APRFC'

if os.path.exists(holding_dir):
	pass
else:
	os.mkdir(holding_dir)

site_numbers = range(1, 494+1) # You can find these from the NWS site
expected_cols = ['River', 'Location', 'Year', 'Breakup Date', 'Ice Moved', 'First Boat',
                     'Un-safe Person', 'Unsafe Vehicle', 'Last Ice', 'Remarks']

def get_driver(executable_path = "/usr/lib/chromium-browser/chromedriver"):
	options = webdriver.ChromeOptions()
	options.add_argument("--headless")
	#service = Service(service)
	return webdriver.Chrome(executable_path=executable_path, options=options)


for idx, site_number in enumerate(site_numbers):
	
	if not os.path.exists(f'{holding_dir}/SITE_{site_number}.pkl'):
		driver = get_driver()
		
		URL_path = f'https://www.weather.gov/aprfc/breakupDB?site={site_number}'
		driver.get(URL_path)
		time.sleep(3) # It needs a chance to sleep
		
		soup = BeautifulSoup(driver.page_source, 'lxml')
		driver.close()

		tables = soup.find_all('table')
		table = tables[-1]
		df = pd.read_html(str(table))

		if len(df[0].columns) == len(expected_cols):
			df[0].to_pickle(f'{holding_dir}/SITE_{site_number}.pkl')

		else:
			
			try:
			
				table = soup.find('table', id = 'datatable')
				df = pd.read_html(str(table))
				
				if len(df[0].columns) == len(expected_cols):
					print('SITE NUMBER:', str(site_number), 'IMPORTED CORRECTLY')
					df[0].to_pickle(f'{holding_dir}/SITE_{site_number}.pkl')

				else:
					print('*******SITE NUMBER:', str(site_number), 'NOT IMPORTING*********')
			except:
				print('SITE NUMBER:', str(site_number), 'NO TABLES TO IMPORT')
				continue
	else:
		continue

		
driver.quit()
os.chdir(holding_dir)
df_list = [pd.read_pickle(i) for i in os.listdir()]
final = pd.concat(df_list)

final['Year'] = final['Year'].astype('int64')
#final = final[final.Year > 1895]
final.sort_values(by=['Year'], inplace = True)
final['Site'] = final['Location'] + ' ' + final['River']

final.to_csv(f'{path_to_breakup_data}/01_BREAK_UP_DATA_WEBSCRAPED.csv')

# Processing the APRFC Data Records

break_up_data = pd.read_csv(f'{path_to_breakup_data}/01_BREAK_UP_DATA_WEBSCRAPED.csv').iloc[:, 1:]

# Removing features that I will not be using
break_up_data.drop(['Ice Moved', 'First Boat', 'Un-safe Person', 'Unsafe Vehicle', 'Last Ice', 'Remarks'], axis = 1, inplace=True)
break_up_data.dropna(inplace=True)

break_up_data['Breakup Date'] = pd.to_datetime(break_up_data['Breakup Date'], format='%Y-%m-%d', errors = 'coerce')
print('There are:', len(break_up_data[break_up_data['Breakup Date'].isna()]), 'dates that are corrupted')

nan_indices = break_up_data.isna().sum(axis=1).astype('bool')

print(break_up_data.loc[nan_indices])

# Editted these Sites as the 
# Breakup Date formats were incorrect (492) or uninterpretable
# (5014)
break_up_data.at[492, 'Breakup Date'] = '1933-05-05'
break_up_data.drop([5014], axis = 0, inplace=True)

break_up_data.reset_index(inplace=True, drop = True)
break_up_data['Breakup Date'] = pd.to_datetime(break_up_data['Breakup Date'], format='%Y-%m-%d')

print()
print('MISSING DATA:', break_up_data.isna().sum())
print(break_up_data.dtypes)

break_up_data.to_pickle(f'{path_to_breakup_data}/01_PROCESSED_BREAK_UP_DATA_WEBSCRAPED.pkl')

end_time = time.time()
complete_time = (end_time - start_time)/60

complete_time = round(complete_time, 2)
print('PROCESS COMPLETE TOOK:', complete_time, 'MINUTES')






