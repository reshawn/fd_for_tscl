import pandas as pd
import numpy as np
from gluonts.dataset.common import ListDataset, TrainDatasets, MetaData
from gluonts.dataset.field_names import FieldName


def series_to_gluonts_dataset(X_train, X_test, params):
    X = pd.concat([X_train, X_test], axis=0)
    value_col = 'target'
    # prep feat dynamic real i.e. other feats aside from target, see https://ts.gluon.ai/stable/tutorials/forecasting/extended_tutorial.html#Use-your-time-series-and-features

    # for gluonts, this first removes values_o as the targe, to separate it from the fields in the gluonts dataset
    # it's pass by ref, so the change carries forward and causes problems
    # instead of copying the entire df and increasing the risk of a memory crash, 
    # save the values_o target col and re-add it later
    target = X[value_col].values
    target_train = X_train[value_col].values

    X.drop(columns=[value_col], inplace=True)
    X_train.drop(columns=[value_col], inplace=True)
    if len(X.columns) == 0:
        fdr = [[]]
        fdr_train = [[]]
    else:
        fdr = np.array(X.T)
        fdr_train = np.array(X_train.T)


    start_timestamp = pd.Timestamp.now()
    train_data = [{
        "start": start_timestamp,
        "target": target_train,
        FieldName.FEAT_DYNAMIC_REAL: fdr_train,
        "item_id": 0
    }]
    test_data = [{
        "start": start_timestamp,
        "target": target,
        FieldName.FEAT_DYNAMIC_REAL: fdr,
        "item_id": 0
    }]

    train_list_dataset = ListDataset(train_data, freq=params['freq'])
    test_list_dataset = ListDataset(test_data, freq=params['freq'])
    metadata = MetaData(
        freq=params['freq'],
        prediction_length=params['prediction_length']
    )
    
    X_train[value_col] = target_train # re-add
    X[value_col] = target # re-add, X is a concat, so shouldnt be an issue here, but just in case im wrong

    dataset = TrainDatasets(metadata=metadata, train=train_list_dataset, test=test_list_dataset)
    return dataset


import ast

def load_params(params_path, key):
    with open(params_path, 'r') as f:
        params = ast.literal_eval(f.read())
    return params[key]