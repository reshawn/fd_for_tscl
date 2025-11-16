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
        adf_result = adfuller(data_chunk) 
        # print(f'{i} p-value={adf_result[1]}, lags={adf_result[2]}')
        num_stat = (num_stat[0], num_stat[1]+1)
        p_values.append(adf_result[1])
        if adf_result[1] < 0.05:
            num_stat = (num_stat[0]+1, num_stat[1])
    # if more than 50% of the p-values are above 0.05, then the data is not stationary
    stationary = num_stat[0] >= num_stat[1]/2
    return stationary

# Loop through monash dir


monash_dir = "monash_data"
results = pd.DataFrame()
for dataset_name in tqdm(os.listdir(monash_dir)):
    print(dataset_name)
    loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(f"{monash_dir}/{dataset_name}")
    dataset = monash_df_to_gluonts_train_datasets(loaded_data, frequency)
    series_lengths = []
    num_stat = 0

    for entry in tqdm(dataset.test):
        row = pd.Series(entry['target'])
        series_lengths.append(len(row))
        stat = run_adf(row)
        if stat: num_stat += 1
    
    results = pd.concat([results, pd.DataFrame([{
        'dataset_name': dataset_name,
        'num_series': len(dataset.test),
        'num_stat': num_stat,
        'pct_stat': num_stat/len(dataset.test),
        'mean_series_len': np.mean(series_lengths),
        'std_series_len': np.std(series_lengths),
        'min_series_len': np.min(series_lengths),
        'max_series_len': np.max(series_lengths),
    }])], ignore_index=True)
    results.to_csv('results_stat_checks.csv')
    
