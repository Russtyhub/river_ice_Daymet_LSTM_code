#!/usr/bin/python3
# conda activate r62GEDI

# This script finds and manually corrects the
# errors that I found in the APRFC database while
# conducting this analysis

import pandas as pd
import numpy as np
from collections import Counter
import pickle

with open("../.directories.pkl", "rb") as file:
    directories = pickle.load(file)

path_to_breakup_data = directories['path_to_breakup_data']
PROCESSED_BREAK_UP_DATA_WEBSCRAPED = pd.read_pickle(f'{path_to_breakup_data}/01_PROCESSED_BREAK_UP_DATA_WEBSCRAPED.pkl')

flag_list = []

for site_name in PROCESSED_BREAK_UP_DATA_WEBSCRAPED.Site.unique():
    site_df = PROCESSED_BREAK_UP_DATA_WEBSCRAPED[PROCESSED_BREAK_UP_DATA_WEBSCRAPED.Site == site_name]

    years = list(site_df['Year'])
    breakup_years = list(site_df['Breakup Date'].dt.year)

    duplicate_years = [item for item, count in Counter(years).items() if count > 1]
    duplicate_breakup_years = [item for item, count in Counter(breakup_years).items() if count > 1]

    off_years = np.equal(np.array(years), np.array(breakup_years))

    if np.product(off_years) == 1:
        pass
    else:
        print(site_name, 'Years column not equal to Breakup date year')
        flag_list.append(1)

    if len(duplicate_years) > 0:
        print(site_name, 'has duplicate years')
        print('DUPLICATE YEARS:', duplicate_years)
        flag_list.append(1)

    if len(duplicate_breakup_years) > 0:
        print(site_name, 'has duplicate breakup date years')
        print('DUPLICATE BREAKUP DATE YEARS:', duplicate_breakup_years)
        flag_list.append(1)
        print('*'*50)

if len(flag_list) == 0:
    print('THERE ARE NO DUPLICATE YEARS IN THE YEARS COLUMN')
    print('NOR DUPLICATE YEARS IN THE BREAKUP DATE COLUMN')

PROCESSED_BREAK_UP_DATA_WEBSCRAPED.loc[(PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Site'] == 
                                        'Alakanuk Yukon River') & 
                                        (PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Year'] == 2024), 'Breakup Date'] = pd.to_datetime('2024-05-24')

PROCESSED_BREAK_UP_DATA_WEBSCRAPED.loc[(PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Site'] == 
                                        'Aniak Kuskokwim River') & 
                                        (PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Year'] == 2023), 'Breakup Date'] = pd.to_datetime('2023-05-15')

PROCESSED_BREAK_UP_DATA_WEBSCRAPED.loc[(PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Site'] == 
                                        'Emmonak Yukon River') & 
                                        (PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Year'] == 2024), 'Breakup Date'] = pd.to_datetime('2024-05-24')

PROCESSED_BREAK_UP_DATA_WEBSCRAPED.loc[(PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Site'] == 
                                        'Fairbanks Chena River') & 
                                        (PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Year'] == 1959), 'Breakup Date'] = pd.to_datetime('1959-05-01')

PROCESSED_BREAK_UP_DATA_WEBSCRAPED.loc[(PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Site'] == 
                                        'Beaver Yukon River') & 
                                        (PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Year'] == 2023), 'Breakup Date'] = pd.to_datetime('2023-05-15')

PROCESSED_BREAK_UP_DATA_WEBSCRAPED.loc[(PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Site'] == 
                                        'Fairbanks D-S of Chena R Tanana River') & 
                                        (PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Year'] == 2024), 'Breakup Date'] = pd.to_datetime('2024-04-30')

PROCESSED_BREAK_UP_DATA_WEBSCRAPED.loc[(PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Site'] == 
                                        'Colville Village Colville River') & 
                                        (PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Year'] == 2024), 'Breakup Date'] = pd.to_datetime('2024-06-08')

PROCESSED_BREAK_UP_DATA_WEBSCRAPED.loc[(PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Site'] == 
                                        'Fairbanks Chena River') & 
                                        (PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Year'] == 1959), 'Breakup Date'] = pd.to_datetime('1959-05-01')

PROCESSED_BREAK_UP_DATA_WEBSCRAPED.loc[(PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Site'] == 
                                        'Emmonak Yukon River') & 
                                        (PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Year'] == 2024), 'Breakup Date'] = pd.to_datetime('2024-05-24')

PROCESSED_BREAK_UP_DATA_WEBSCRAPED.loc[(PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Site'] == 
                                        'Aniak Kuskokwim River') & 
                                        (PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Year'] == 2023), 'Breakup Date'] = pd.to_datetime('2023-05-15')

PROCESSED_BREAK_UP_DATA_WEBSCRAPED.loc[(PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Site'] == 
                                        'Beaver Yukon River') & 
                                        (PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Year'] == 2023), 'Breakup Date'] = pd.to_datetime('2023-05-15')

PROCESSED_BREAK_UP_DATA_WEBSCRAPED.loc[(PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Site'] == 
                                        'Alakanuk Yukon River') & 
                                        (PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Year'] == 2024), 'Breakup Date'] = pd.to_datetime('2024-05-24')

PROCESSED_BREAK_UP_DATA_WEBSCRAPED.loc[(PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Site'] == 
                                        'Fairbanks D-S of Chena R Tanana River') & 
                                        (PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Year'] == 2024), 'Breakup Date'] = pd.to_datetime('2024-04-30')

PROCESSED_BREAK_UP_DATA_WEBSCRAPED.loc[(PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Site'] == 
                                        'Colville Village Colville River') & 
                                        (PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Year'] == 2024), 'Breakup Date'] = pd.to_datetime('2024-06-08')

# this record was formatted incorrectly. It was accounted for at the bottom of
# 01_Webscraper_Break-up_Data.py because it had to be fixed ASAP to allow the dates
# to be read as dates. This placeholder is to keep the information together. Incidentally
# record 5014 had to be removed from the original df as it was unusable. 
PROCESSED_BREAK_UP_DATA_WEBSCRAPED.loc[(PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Site'] == 
                                        'Nenana Tanana River') & 
                                        (PROCESSED_BREAK_UP_DATA_WEBSCRAPED['Year'] == 1935), 'Breakup Date'] = pd.to_datetime('1935-05-15')

PROCESSED_BREAK_UP_DATA_WEBSCRAPED.to_pickle(f'{path_to_breakup_data}/01_PROCESSED_BREAK_UP_DATA_WEBSCRAPED.pkl')



