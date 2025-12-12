# %pip install '/content/chronos-forecasting-main.zip' 'pandas[pyarrow]' 'matplotlib' 'gluonts' 'mxnet' # git+https://github.com/amazon-science/chronos-forecasting@main

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import mean_absolute_error

def eval_mase(y, y_hat, pl, seasonality):
  '''
  y:  np.array, original series of testset including forecasted values
  yhat: np.array, forecasted values
  pl: int, prediction length
  seasonality: int, seasonality of the time series
  '''
  training_data = y[:-pl]
  ground_truth = y[-pl:]

  y_pred_naive = np.array(training_data)[:-int(seasonality)]
  mae_naive = mean_absolute_error(np.array(training_data)[int(seasonality):], y_pred_naive, multioutput="uniform_average")

  mae_score = mean_absolute_error(
      ground_truth,
      y_hat,
      sample_weight=None,
      multioutput="uniform_average",
  )

  epsilon = np.finfo(np.float64).eps

  mase_score = mae_score / np.maximum(mae_naive, epsilon)
  return mase_score

def get_weights_ffd(diff_amt, thresh, lim):
    weights = [1.]
    k = 1
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
    weights = np.array(weights[::-1]).reshape(-1, 1)
    return weights

def invert_ffd(new_fd_series, original_series, d, thresh, full_original=False):
    # reminder that original_series means training_series without ground truth of pred, although it should overwrite
    original_series.reset_index(drop=True, inplace=True) # reset index to ensure regular integer index for easier slicing
    weights = get_weights_ffd(d, thresh, len(original_series))
    width = len(weights) - 1 # -1 to allow for slicing a window of len(weights)
    new_unfd_series = original_series.copy()
    new_fd_series = new_fd_series.copy()
    if full_original:
        new_fd_series.index += width # shift the index to align with the original series index, i.e. before nans were dropped and the index reset on the frac diffed series

    for i in range(original_series.index[-1]+1, new_fd_series.index[-1]+1): # weirdly taking the last index + 1 instead of len/shape because of the index shift on the new fd series
        new_unfd_series.loc[i] = (new_fd_series.loc[i] - np.dot(weights[:-1].T, new_unfd_series[i-width:i]))[0]
    return new_unfd_series


def pred_eval(context_df, pipeline, pl, quantile_levels=[0.1, 0.5, 0.9], seasonality=288, d=None, thresh=None, item_id='series_0'):

  def invert_pred(pred_df, context_df, d, thresh):
    training_context = context_df[:-pl]
    full_fd_series_pred = pd.concat([training_context[['target']], pred_df[['predictions']]])
    full_unfd_series = invert_ffd(full_fd_series_pred, training_context['target_o'], d, thresh)
    return full_unfd_series[-pl:]

  # input df, default target col str is "target" other columns are taken as past covariates

  context_df.reset_index(inplace=True, drop=True)
  # doesnt seem to handle missing easily, so replace with incorrect dates with correct frequency
  context_df['timestamp'] = pd.date_range(start='2003-10-17 19:15:00', freq="5min", periods=len(context_df))
  context_df['item_id'] = item_id
  covariates = context_df.columns.tolist()
  covariates.remove('target')

  train_inputs = []
  for item_id, group in context_df[:4000].groupby("item_id"):
      train_inputs.append({
          "target": group['target'].values,
          "past_covariates": {col: group[col].values for col in covariates},
      })
  finetuned_pipeline = pipeline.fit(
      inputs=train_inputs,
      prediction_length=pl,
      num_steps=2_000,  
      finetune_mode='lora',
      learning_rate=1e-5,
      batch_size=32,
      logging_steps=100,
  )

  pred_df = finetuned_pipeline.predict_df(context_df, prediction_length=pl, quantile_levels=quantile_levels)

  y_hat = pred_df[['predictions']].to_numpy()
  y = context_df[['target']].to_numpy()

  if d is not None and thresh is not None:
    y_hat = invert_pred(pred_df, context_df, d, thresh)
    y = context_df[['target_o']].to_numpy()

  mase = eval_mase(y, y_hat, pl=pl, seasonality=seasonality)
  return mase, pred_df

# dict to txt

import json

def save(results, filename):
    with open(f'{filename}.txt', 'w') as f:
        json.dump(results, f)

import pickle

with open('spy5m_bintp_labelled.pkl', 'rb') as f:
    df_original = pickle.load(f) # ohlv + transactions + labels + bintp labels

# with open('spy5m_fd.pkl', 'rb') as f:
#     df_fd = pickle.load(f) # d=0.25 when using this one, post h=l drop
with open('spy5m_bintp004_episodes_fracdiff.pkl', 'rb') as f:
    df_fd = pickle.load(f) # fracdiffed ohlcv + transactions + labels
# with open('/mnt/c/Users/resha/Documents/Github/balancing_framework/spy5m_labelled_episodes_ta.pkl', 'rb') as f:
#     df_ta = pickle.load(f) # ohlcv + ~120 TA features + labels
# with open('/mnt/c/Users/resha/Documents/Github/balancing_framework/spy5m_ta_fracdiff.pkl', 'rb') as f:
#     df_fd_ta = pickle.load(f) # fracdiffed ohlcv + transactions + ~120 TA features + labels
# with open('spy5m_smallta.pkl', 'rb') as f:
#     df_ta = pickle.load(f)
# with open('spy5m_smallta_fracdiff.pkl', 'rb') as f:
#     df_fd = pickle.load(f)

df_original = df_original.loc['01-01-2018':]
df_fd = df_fd.loc['01-01-2018':]

# filter single order bars
# df_original.drop(df_original[df_original['high'] == df_original['low']].index, inplace=True)
# df_fd.drop(df_fd[df_fd['high'] == df_fd['low']].index, inplace=True)

# df_original.drop(df_original[df_original['transactions'] == 1].index, inplace=True)
# df_fd.drop(df_fd[df_fd['transactions'] == 1].index, inplace=True)


d = 0.2 # 0.2 0.25
thresh = 1e-5

df_size = 20_000
# original series target with ohlcv
df = df_original[["volume", "vwap", "open", "close", "high", "low", "transactions"]] # 0.01 0.001
end_ts = df_original[:df_size].index[-1]
X_withoutfd= df.loc[:end_ts].copy()
X_withoutfd['target'] = X_withoutfd['close'].shift(-1)
X_withoutfd.dropna(inplace=True)

# fd target with fd cols and original cols
df2 = df_fd[["volume", "vwap", "open", "close", "high", "low", "transactions"]]
X_fdtarget = df2.loc[:end_ts].copy()
X_fdtarget = X_fdtarget.join(X_withoutfd.add_suffix(f'_o'), how='outer')
X_fdtarget['target'] = X_fdtarget['close'].shift(-1)
X_fdtarget.dropna(inplace=True)

# switch above to have original target with fd in supp
X_otarget_fdinfdr = X_fdtarget.copy()
X_otarget_fdinfdr.rename(columns={"target": "target_fd", "target_o": "target"}, inplace=True)

del df2, df, df_fd, df_original





# Use only 1 GPU if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from chronos import BaseChronosPipeline, Chronos2Pipeline

# Load the Chronos-2 pipeline
# GPU recommended for faster inference, but CPU is also supported
pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda")


pl=78
results = []
dataset = 'SP500'
filename = 'sp500_finetuned_dropped_2ksteps_2018_4ktuning'

# mase, pred_df = pred_eval(X_withoutfd, pipeline, pl, item_id=dataset)
# form = 'original'
# print(f'{dataset} {form} pl {pl}, mase: {mase:.4f}')
# results.append({'dataset':dataset, 'form':form, 'pl':pl, 'mase':mase})
# save(results, filename)

# mase, pred_df = pred_eval(X_otarget_fdinfdr, pipeline, pl, item_id=dataset)
# form = 'original fd supp'
# print(f'{dataset} {form} pl {pl}, mase: {mase:.4f}')
# results.append({'dataset':dataset, 'form':form, 'pl':pl, 'mase':mase})
# save(results, filename)

# mase, pred_df = pred_eval(X_fdtarget, pipeline, pl, d=d, thresh=thresh, item_id=dataset)
# form = 'fd'
# print(f'{dataset} {form} pl {pl}, mase: {mase:.4f}')
# results.append({'dataset':dataset, 'form':form, 'pl':pl, 'mase':mase})
# save(results, filename)

# pl=12

# mase, pred_df = pred_eval(X_withoutfd, pipeline, pl, item_id=dataset)
# form = 'original'
# print(f'{dataset} {form} pl {pl}, mase: {mase:.4f}')
# results.append({'dataset':dataset, 'form':form, 'pl':pl, 'mase':mase})
# save(results, filename)


# mase, pred_df = pred_eval(X_otarget_fdinfdr, pipeline, pl, item_id=dataset)
# form = 'original fd supp'
# print(f'{dataset} {form} pl {pl}, mase: {mase:.4f}')
# results.append({'dataset':dataset, 'form':form, 'pl':pl, 'mase':mase})
# save(results, filename)

# mase, pred_df = pred_eval(X_fdtarget, pipeline, pl, d=d, thresh=thresh, item_id=dataset)
# form = 'fd'
# print(f'{dataset} {form} pl {pl}, mase: {mase:.4f}')
# results.append({'dataset':dataset, 'form':form, 'pl':pl, 'mase':mase})
# save(results, filename)

# for comp to finetuned
pl=780

mase, pred_df = pred_eval(X_withoutfd, pipeline, pl, item_id=dataset)
form = 'original'
print(f'{dataset} {form} pl {pl}, mase: {mase:.4f}')
results.append({'dataset':dataset, 'form':form, 'pl':pl, 'mase':mase})
save(results, filename)

mase, pred_df = pred_eval(X_otarget_fdinfdr, pipeline, pl, item_id=dataset)
form = 'original fd supp'
print(f'{dataset} {form} pl {pl}, mase: {mase:.4f}')
results.append({'dataset':dataset, 'form':form, 'pl':pl, 'mase':mase})
save(results, filename)

mase, pred_df = pred_eval(X_fdtarget, pipeline, pl, d=d, thresh=thresh, item_id=dataset)
form = 'fd'
print(f'{dataset} {form} pl {pl}, mase: {mase:.4f}')
results.append({'dataset':dataset, 'form':form, 'pl':pl, 'mase':mase})
save(results, filename)

print(results)
# import pickle

# X = pd.read_csv('m4_1165_ffd.csv') # fixed width fd, 0.01 thresh, d=0.95
# X.rename(columns={"values_o": "target_o", "values_fd": "target_fd"}, inplace=True)
# dataset = 'm4_daily_dataset'
# d=0.95
# thresh = 0.01


# X_withoutfd = X.drop(columns=['target_fd']).rename(columns={"target_o":"target"})
# X_fdtarget = X.rename(columns={"target_fd": "target"})
# # this X_fdtarget leaves o in fdr, fd_fdr is still the opposite to compare the impact of fd as target,
# # and so far most tests have shown o+fd being more effective than fd alone
# X_otarget_fdinfdr = X.rename(columns={"target_o": "target"})


# pl=14
# results = []
# dataset = 'm4_1165'

# mase, pred_df = pred_eval(X_withoutfd, pipeline, pl)
# form = 'original'
# print(f'{dataset} {form} pl {pl}, mase: {mase:.4f}')
# results.append({'dataset':dataset, 'form':form, 'pl':pl, 'mase':mase})

# mase, pred_df = pred_eval(X_otarget_fdinfdr, pipeline, pl)
# form = 'original fd supp'
# print(f'{dataset} {form} pl {pl}, mase: {mase:.4f}')
# results.append({'dataset':dataset, 'form':form, 'pl':pl, 'mase':mase})

# mase, pred_df = pred_eval(X_fdtarget, pipeline, pl, d=0.95, thresh=0.01)
# form = 'fd'
# print(f'{dataset} {form} pl {pl}, mase: {mase:.4f}')
# results.append({'dataset':dataset, 'form':form, 'pl':pl, 'mase':mase})

# pl=140

# mase, pred_df = pred_eval(X_withoutfd, pipeline, pl)
# form = 'original'
# print(f'{dataset} {form} pl {pl}, mase: {mase:.4f}')
# results.append({'dataset':dataset, 'form':form, 'pl':pl, 'mase':mase})

# mase, pred_df = pred_eval(X_otarget_fdinfdr, pipeline, pl)
# form = 'original fd supp'
# print(f'{dataset} {form} pl {pl}, mase: {mase:.4f}')
# results.append({'dataset':dataset, 'form':form, 'pl':pl, 'mase':mase})

# mase, pred_df = pred_eval(X_fdtarget, pipeline, pl, d=0.95, thresh=0.01)
# form = 'fd'
# print(f'{dataset} {form} pl {pl}, mase: {mase:.4f}')
# results.append({'dataset':dataset, 'form':form, 'pl':pl, 'mase':mase})

# # for comp to finetuned
# pl=700

# mase, pred_df = pred_eval(X_withoutfd, pipeline, pl)
# form = 'original'
# print(f'{dataset} {form} pl {pl}, mase: {mase:.4f}')
# results.append({'dataset':dataset, 'form':form, 'pl':pl, 'mase':mase})

# mase, pred_df = pred_eval(X_otarget_fdinfdr, pipeline, pl)
# form = 'original fd supp'
# print(f'{dataset} {form} pl {pl}, mase: {mase:.4f}')
# results.append({'dataset':dataset, 'form':form, 'pl':pl, 'mase':mase})

# mase, pred_df = pred_eval(X_fdtarget, pipeline, pl, d=0.95, thresh=0.01)
# form = 'fd'
# print(f'{dataset} {form} pl {pl}, mase: {mase:.4f}')
# results.append({'dataset':dataset, 'form':form, 'pl':pl, 'mase':mase})

# save(results, 'm4_finetuned_2k')