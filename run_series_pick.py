# re run checks to store more on the series 

from gluonts.dataset.repository import get_dataset, dataset_names
from gluonts.dataset.util import to_pandas

import pickle
import pandas as pd
import numpy as np
import json
import time
import argparse
from tqdm import tqdm

from wrappers import run_measurements, viz
from fracdiff import frac_diff_bestd
from monash_data_utils import convert_tsf_to_dataframe, monash_df_to_gluonts_train_datasets
import os

from statsmodels.tsa.stattools import adfuller

def run_adf(series):
    adf_chunk_size = 100_000
    num_stat = (0,0) # number of stationary windows, total number of windows
    p_values = []
    for i in range(0, len(series), adf_chunk_size):
        data_chunk = series.dropna()[i:i+adf_chunk_size]
        if data_chunk.nunique()==1:
            print(f'series has only one unique value: {series.iloc[0]}')
            continue
        try:
            adf_result = adfuller(data_chunk) 
        except Exception as e:
            print(e)
            continue
        # print(f'{i} p-value={adf_result[1]}, lags={adf_result[2]}')
        num_stat = (num_stat[0], num_stat[1]+1)
        p_values.append(adf_result[1])
        if adf_result[1] < 0.05:
            num_stat = (num_stat[0]+1, num_stat[1])
    # if more than 50% of the p-values are above 0.05, then the data is not stationary
    stationary = num_stat[0] >= num_stat[1]/2
    return stationary, np.mean(p_values)

# Loop through monash dir


monash_dir = "monash_data"
results = pd.DataFrame()
for dataset_name in tqdm(os.listdir(monash_dir)):
    print(dataset_name)
    loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(f"{monash_dir}/{dataset_name}")
    dataset = monash_df_to_gluonts_train_datasets(loaded_data, frequency)

    for index, entry in tqdm(enumerate(dataset.test)):
        row = pd.Series(entry['target'])
        stat, pva = run_adf(row)
        if not stat: 
            new_row = pd.DataFrame([{
                'dataset_name': dataset_name,
                'series_idx': index,
                'avg_pvalue': pva,
                'series_length': len(row),
            }])
            new_row.to_csv('series_pick_checks.csv', mode='a', index=False, header=False)
    

#
results = pd.DataFrame()
for dataset_name in tqdm(dataset_names):
    try:
        dataset = get_dataset(dataset_name)
        print(dataset_name)
    except Exception as e:
        print(e)
        continue

    for index, entry in enumerate(dataset.test):
        row = pd.Series(entry['target'])
        stat, pva = run_adf(row)
        if not stat: 
            new_row = pd.DataFrame([{
                'dataset_name': dataset_name,
                'series_idx': index,
                'avg_pvalue': pva,
                'series_length': len(row),
            }])
            new_row.to_csv('series_pick_checks_gluonts.csv', mode='a', index=False, header=False)
    




