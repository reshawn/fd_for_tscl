
import numpy as np
import pandas as pd



def get_weights(diff_amt, size):
    """
    :param diff_amt: (float) Differencing amount
    :param size: (int) Length of the series
    :return: (np.ndarray) Weight vector
    """

    # The algorithm below executes the iterative estimation (section 5.4.2, page 78)
    weights = [1.]  # create an empty list and initialize the first element with 1.
    for k in range(1, size):
        weights_ = -weights[-1] * (diff_amt - k + 1) / k  # compute the next weight
        weights.append(weights_)

    # Now, reverse the list, convert into a numpy column vector
    weights = np.array(weights[::-1]).reshape(-1, 1)
    return weights

def frac_diff(series, diff_amt, thresh=0.01):
    """
    :param series: (pd.Series) A time series that needs to be differenced
    :param diff_amt: (float) Differencing amount
    :param thresh: (float) Threshold or epsilon
    :return: (pd.DataFrame) Differenced series
    """

    # 1. Compute weights for the longest series
    weights = get_weights(diff_amt, series.shape[0])

    # 2. Determine initial calculations to be skipped based on weight-loss threshold
    # added - ignoring skips to retain data
    # measures total contribution of weights by taking each as a percentage of the accumulated sum
    weights_ = np.cumsum(abs(weights))
    weights_ /= weights_[-1]
    skip = weights_[weights_ > thresh].shape[0] # total number of weights above the threshold
    # setting skip to this means we ensure we have at least this number of high contributing weights from the start of the new series
    print(f'Skipping {skip} rows for this d={diff_amt}')
    # skip = 0

    # 3. Apply weights to values
    output_df = {}
    for name in series.columns:
        series_f = series[[name]].fillna(method='ffill').dropna()
        output_df_ = pd.Series(index=series.index, dtype='float64')

        for iloc in range(skip, series_f.shape[0]):
            loc = series_f.index[iloc]

            # At this point all entries are non-NAs so no need for the following check
            # if np.isfinite(series.loc[loc, name]):
            output_df_[loc] = np.dot(weights[-(iloc + 1):, :].T, series_f.loc[:loc])[0, 0]

        output_df[name] = output_df_.copy(deep=True)
    output_df = pd.concat(output_df, axis=1)
    return output_df

def get_weights_ffd(diff_amt, thresh, lim):
    """
    :param diff_amt: (float) Differencing amount
    :param thresh: (float) Threshold for minimum weight
    :param lim: (int) Maximum length of the weight vector
    :return: (np.ndarray) Weight vector
    """

    weights = [1.]
    k = 1

    # The algorithm below executes the iterativetive estimation (section 5.4.2, page 78)
    # The output weights array is of the indicated length (specified by lim)
    ctr = 0
    while True:
        # compute the next weight
        weights_ = -weights[-1] * (diff_amt - k + 1) / k

        if abs(weights_) < thresh:
            break

        weights.append(weights_)
        k += 1
        ctr += 1
        if ctr == lim - 1:  # if we have reached the size limit, exit the loop
            break

    # Now, reverse the list, convert into a numpy column vector
    weights = np.array(weights[::-1]).reshape(-1, 1)
    return weights

def frac_diff_ffd(series, diff_amt, thresh=1e-5):
    """
    :param series: (pd.Series) A time series that needs to be differenced
    :param diff_amt: (float) Differencing amount
    :param thresh: (float) Threshold for minimum weight
    :return: (pd.DataFrame) A data frame of differenced series
    """

    # 1) Compute weights for the longest series
    weights = get_weights_ffd(diff_amt, thresh, series.shape[0])
    width = len(weights) - 1

    # 2) Apply weights to values
    # 2.1) Start by creating a dictionary to hold all the fractionally differenced series
    output_df = {}

    # 2.2) compute fractionally differenced series for each stock
    for name in series.columns:
        series_f = series[[name]].fillna(method='ffill').dropna()
        temp_df_ = pd.Series(index=series.index, dtype='float64')
        for iloc1 in range(width, series_f.shape[0]):
            loc0 = series_f.index[iloc1 - width]
            loc1 = series.index[iloc1]

            # At this point all entries are non-NAs, hence no need for the following check
            # if np.isfinite(series.loc[loc1, name]):
            temp_df_[loc1] = np.dot(weights.T, series_f.loc[loc0:loc1])[0, 0]

        output_df[name] = temp_df_.copy(deep=True)

    # transform the dictionary into a data frame
    output_df = pd.concat(output_df, axis=1)
    return output_df

def invert_fd(new_fd_series, original_series, d):
    # frac diff iteratively appies the most recent n weights to the n values of the series so far to produce the transformed series
    # so to reverse it, we cant reverse the entire dot product operation of the final series, but can iteratively solve for one unknown, the newest unfracdiffed value
    # for that, we subtract the dot product of the previous values from the current frac diff value, and divide by the current weight
    # i.e. if a = weights where len(weights) = n+1 and n is the length of the known original series
    # b = original_series of length n
    # c = frac diff series of length n+1 including the new to be inverted value
    # then the new value to be added to b = (c - np.dot(a[:-1], b[:-1])) / a[-1] 
    # or (the current frac diff value - the 1 less than complete frac diff calculation for this current value) / the current weight
    # and the current weight will always be 1 as the first in that series
    weights = get_weights(d, len(new_fd_series))
    new_unfd_series = original_series.copy()
    for i in range(len(original_series), len(new_fd_series)):
        # new_unfd_series will have all known values so far up to i-1 as the index of the final inverted series
        # the slice of the weights for the incomplete calc would be the last len(new_unfd_series) weights + 1 to shift back for the current weight we divide by, which is always 1
        new_unfd_series.loc[i] = (new_fd_series.loc[i] - np.dot(weights[-(len(new_unfd_series)+1):-1, :].T, new_unfd_series))[0]
    return new_unfd_series

def invert_ffd(new_fd_series, original_series, d, thresh, full_original=False):
    # *********** 
    # NOTE: full_original assumes that the original series is the actual original series before the FD was applied.
    # NOT the re-aligned original series after FD where initial rows are dropped. The width adjustment of the index does the re-alignment in this function
    # ***********
    # similar as the original invert_fd function,
    # or (the current frac diff value - the 1 less than complete frac diff calculation for this current value) / the current weight
    # and the current weight will always be 1 as the first in that series
    # so the difference is in the 1 less than complete calc, which would be sliced to a window length
    # that window gets decided by the thresh and series length values used
    original_series.reset_index(drop=True, inplace=True) # reset index to ensure regular integer index for easier slicing
    weights = get_weights_ffd(d, thresh, len(original_series))
    width = len(weights) - 1 # -1 to allow for slicing a window of len(weights)
    new_unfd_series = original_series.copy()
    new_fd_series = new_fd_series.copy()
    if full_original:
        new_fd_series.index += width # shift the index to align with the original series index, i.e. before nans were dropped and the index reset on the frac diffed series

    for i in range(original_series.index[-1]+1, new_fd_series.index[-1]+1): # weirdly taking the last index + 1 instead of len/shape because of the index shift on the new fd series
        # new_unfd_series will have all known values so far up to i-1 as the index of the final inverted series
        # the slice of the weights for the incomplete calc would be the last len(new_unfd_series) weights + 1 to shift back for the current weight we divide by, which is always 1
        new_unfd_series.loc[i] = (new_fd_series.loc[i] - np.dot(weights[:-1].T, new_unfd_series[i-width:i]))[0]
    return new_unfd_series

from statsmodels.tsa.stattools import adfuller
def set_thresh(series, diff_amt, max_rows_removed_ratio):
    # should use a smaller thresh and increase to decrease the rows dropped, since a bigger thresh means smaller fd window
    # starting point could be a param too, but right now the small windows are of some interest for testing
    thresh = 0.01  # start with a relatively large thresh value
    while True:
        weights =  get_weights_ffd(diff_amt, thresh, len(series)) # get_weights_ffd(diff_amt, thresh, len(series))
        rows_removed = len(weights) - 1
        if rows_removed / len(series) <= max_rows_removed_ratio:
            break
        thresh *= 2  # increase thresh
    return thresh
def frac_diff_bestd(df, type='fd'):
    saved_params = {}
    d_tests = np.arange(0,1,0.05)
    changed = 0
    # for col in tqdm(df.columns): # more verbose
    for col in df.columns:
        thresh = None
        print(col)
        if df[col].nunique()==1:
            print(f'{col} has only one unique value: {df[col].iloc[0]}')
            continue
        for d in d_tests:
            if type=='ffd':
                # thresh = set_thresh(df[col], d, max_rows_removed_ratio=0.25) # using this leads to significantly different results, 
                # but test with it more if the dropped rows becomes a problem again
                # remember to consider the starting point, and the series in focus
                # a picked non stat series in m4 that benefitted from the short window more is the reason why those are still a point of interest
                # despite the windowing being more of an efficiency addition
                frac_diff_test = frac_diff_ffd(df[[col]], d ) # , thresh
            else:
                frac_diff_test = frac_diff(df[[col]], d  )
            frac_diff_test.dropna(inplace=True)
            # at the time of writing, series length 885k crashes the kernel
            # 885k rows is too much for adf with current memory limits, options: 1. use just the first 100k rows,
            # 2. changing the maxlag or autolag params, 3. using a rolling window agg of the data, 4. testing the whole dataset in chunks of 100k
            
            # option 1:
            # sample_size = 100000  # Adjust based on your testing capacity
            # data_sample = frac_diff_test[col].dropna()[:sample_size]
            # adf_result = adfuller(data_sample) 
            # print(f'{col} d={d} p-value={adf_result[1]}')
            # if adf_result[1] < 0.05:

            # option 4:
            adf_chunk_size = 100_000
            num_stat = (0,0) # number of stationary windows, total number of windows
            p_values = []
            for i in range(0, len(frac_diff_test[col]), adf_chunk_size):
                data_chunk = frac_diff_test[col][i:i+adf_chunk_size]
                if data_chunk.nunique()<=1:
                    print(f'{col} has only one unique value in adf chunking: {frac_diff_test[col].iloc[0]}')
                    continue
                adf_result = adfuller(data_chunk) 
                # print(f'{i} p-value={adf_result[1]}, lags={adf_result[2]}')
                num_stat = (num_stat[0], num_stat[1]+1)
                p_values.append(adf_result[1])
                if adf_result[1] < 0.05:
                    num_stat = (num_stat[0]+1, num_stat[1])
            # if more than 50% of the p-values are above 0.05, then the data is not stationary
            stationary = num_stat[0] >= num_stat[1]/2
            print(f"{col} d={d} stat windows ={num_stat[0]} out of {num_stat[1]} p-values = {p_values}")

            if stationary:
                # stationary with this d value
                df[col] = frac_diff_test[col]
                # print(f'{col} stationary with d={d} p-value={adf_result[1]}')
                print(f'{col} stationary with d={d} thresh={thresh} stat windows ={num_stat[0]} out of {num_stat[1]} p-values = {p_values}')

                if d != 0:
                    changed += 1
                break
        saved_params[col] = (d, thresh)
    print(f'changed {changed} out of {len(df.columns)} columns; {changed/len(df.columns)*100}%')
    return df, (changed/len(df.columns))*100, saved_params