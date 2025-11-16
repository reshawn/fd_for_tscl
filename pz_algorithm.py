import pickle
import pandas as pd 
import numpy as np


def get_local_extrema(series, extrema):
    '''
    Get local maxima or minima of a series.
    A maxima or peak is defined as a point where the current value is greater than the values directly preceding and succeeding it.
    In the case of consecutive equal values, the current value is compared to the last or next different value.
    i.e. if today's price is greater than yesterday's and tomorrow's price, then today is a peak.
    similarly, if today's price is less than yesterday's and tomorrow's price, then today is a trough.
    series: pd.Series with datetime index
    extrema: 'max' or 'min'
    returns: pd.Series with datetime index
    '''
    # these first lines cover local maxima or minima where there are no back to back equal prices
    if extrema=='max':
        subset = series[(series.shift(1) < series) & (series.shift(-1) < series)] # shift 1 is prior shift -1 is next
    elif extrema=='min':
        subset = series[(series.shift(1) > series) & (series.shift(-1) > series)]

    # we iterate to find local maxima and minima where there are back to back equal values
    # i.e. "When two or more daily closing prices in a row are equal, 
    # the last of them is compared with the prices directly preceding and succeeding the string of equal prices."
    for i, current in enumerate(series):
        if i == 0:
            continue
        if i == len(series) - 1:
            continue

        if series.iloc[i-1] != current and series.iloc[i+1] != current:
            continue

        prior = series.iloc[i-1]
        next = series.iloc[i+1]
        step = 1
        while prior == current:
            step += 1
            prior = series.iloc[i-step]
        step = 1
        while next == current:
            step += 1
            next = series.iloc[i+step]
        
        if extrema=='max':
            if prior < current > next:
                subset = pd.concat([subset,pd.Series(current, index=[series.index[i]])])
        elif extrema=='min':
            if prior > current < next:
                subset = pd.concat([subset,pd.Series(current, index=[series.index[i]])])
            
    subset.sort_index(inplace=True)
    return subset



def drop_consecutive_extrema(peaks, troughs):
    '''
    Checks for consecutive peaks with no troughs between them, and removes the lower one.
    Similarly for consecutive troughs with no peaks between them and removes the higher one.
    peaks: pd.Series with datetime index
    troughs: pd.Series with datetime index
    returns: Tuple[pd.Series, pd.Series]; peaks, troughs
    '''
    peaks_df = pd.DataFrame(peaks, columns=['price'])
    peaks_df['extrema'] = 'max'
    troughs_df = pd.DataFrame(troughs, columns=['price'])
    troughs_df['extrema'] = 'min'
    extrema_df = pd.concat([peaks_df, troughs_df])
    extrema_df.sort_index(inplace=True)

    # check for consecutive peaks with no troughs between them, and remove the lower one
    peaks_todrop = []
    troughs_todrop = []
    for i, row in extrema_df.iterrows():
        if i == extrema_df.index[0]:
            last_row = row
            continue
        if row['extrema'] == 'max' and last_row['extrema'] == 'max':
            if row['price'] < last_row['price']:
                peaks_todrop.append(i)
            else:
                peaks_todrop.append(last_row.name)
        elif row['extrema'] == 'min' and last_row['extrema'] == 'min':
            if row['price'] > last_row['price']:
                troughs_todrop.append(i)
            else:
                troughs_todrop.append(last_row.name)
        last_row = row

    peaks = peaks.drop(peaks_todrop)
    troughs = troughs.drop(troughs_todrop)
    return peaks, troughs

def pz_algorithm(series, iterations=3):
    '''
    Find peaks and troughs using the PZ algorithm.
    series: pd.Series with datetime index
    returns: Tuple[pd.Series, pd.Series]; peaks, troughs
    '''
    # define the initial set of peaks and troughs
    peaks = get_local_extrema(series, 'max')
    troughs = get_local_extrema(series, 'min')

    # repeat for the number of specified iterations
    for i in range(iterations):
        peaks = get_local_extrema(peaks, 'max')
        troughs = get_local_extrema(troughs, 'min')
        # check for consecutive peaks with no troughs between them, and remove the lower one
        peaks, troughs = drop_consecutive_extrema(peaks, troughs)
    return peaks, troughs