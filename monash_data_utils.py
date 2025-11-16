
# THE FOLLOWING IS FOR LOADING TSF DATASETS AS GLUONTS DATASETS FROM FILES

# tsf data loader

# Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the dataset: frequency, horizon, whether the dataset contains missing values and whether the series have equal lengths
#
# Parameters
# full_file_path_and_name - complete .tsf file path
# replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
# value_column_name - Any name that is preferred to have as the name of the column containing series values in the returning dataframe

# Example of usage
# loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe("TSForecasting/tsf_data/sample.tsf")

# print(loaded_data)
# print(frequency)
# print(forecast_horizon)
# print(contain_missing_values)
# print(contain_equal_length)

from distutils.util import strtobool
from datetime import datetime
import pandas as pd
from gluonts.dataset.field_names import FieldName

def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(  # type:ignore   not my code
                            pd.Series(numeric_series).array
                        )

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series  # type:ignore   not my code
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


def get_dataset_from_file(dataset_name, external_forecast_horizon, context_length):
    (
        df,
        frequency,
        forecast_horizon,
        contain_missing_values,
        contain_equal_length,
    ) = convert_tsf_to_dataframe(f"{dataset_name}.tsf", "NaN", "series_value")

    VALUE_COL_NAME = "series_value"
    TIME_COL_NAME = "start_timestamp"
    SEASONALITY_MAP = {
        "minutely": [1440, 10080, 525960],
        "10_minutes": [144, 1008, 52596],
        "half_hourly": [48, 336, 17532],
        "hourly": [24, 168, 8766],
        "daily": 7,
        "weekly": 365.25 / 7,
        "monthly": 12,
        "quarterly": 4,
        "yearly": 1,
    }
    FREQUENCY_MAP = {
        "minutely": "1min",
        "10_minutes": "10min",
        "half_hourly": "30min",
        "hourly": "1H",
        "daily": "1D",
        "weekly": "1W",
        "monthly": "1M",
        "quarterly": "1Q",
        "yearly": "1Y",
    }

    train_series_list = []
    test_series_list = []
    train_series_full_list = []
    test_series_full_list = []
    final_forecasts = []

    if frequency is not None:
        freq = FREQUENCY_MAP[frequency]
        seasonality = SEASONALITY_MAP[frequency]
    else:
        freq = "1Y"
        seasonality = 1

    if isinstance(seasonality, list):
        seasonality = min(seasonality)  # Use to calculate MASE

    # If the forecast horizon is not given within the .tsf file, then it should be provided as a function input
    if forecast_horizon is None:
        if external_forecast_horizon is None:
            raise Exception("Please provide the required forecast horizon")
        else:
            forecast_horizon = external_forecast_horizon

    start_exec_time = datetime.now()

    for index, row in df.iterrows():
        if TIME_COL_NAME in df.columns:
            train_start_time = row[TIME_COL_NAME]
        else:
            train_start_time = datetime.strptime(
                "1900-01-01 00-00-00", "%Y-%m-%d %H-%M-%S"
            )  # Adding a dummy timestamp, if the timestamps are not available in the dataset or consider_time is False

        series_data = row[VALUE_COL_NAME]

        # Creating training and test series. Test series will be only used during evaluation
        train_series_data = series_data[: len(series_data) - forecast_horizon]
        test_series_data = series_data[
            (len(series_data) - forecast_horizon) : len(series_data)
        ]

        train_series_list.append(train_series_data)
        test_series_list.append(test_series_data)

        # We use full length training series to train the model as we do not tune hyperparameters
        train_series_full_list.append(
            {
                FieldName.TARGET: train_series_data,
                FieldName.START: pd.Timestamp(train_start_time),  # freq=freq),
            }
        )

        test_series_full_list.append(
            {
                FieldName.TARGET: series_data,
                FieldName.START: pd.Timestamp(train_start_time),  # freq=freq),
            }
        )

    train_ds = ListDataset(train_series_full_list, freq=freq)  # type:ignore not my code
    test_ds = ListDataset(test_series_full_list, freq=freq)  # type:ignore not my code

    return train_ds, test_ds, freq, seasonality

GLUONTS_FREQ_MAP = {
    # Time-based frequencies
    'seconds': 'S',
    '4_seconds': '4S',
    '15_seconds': '15S',
    '30_seconds': '30S',
    
    'minute': 'T',
    '5_minutes': '5T',
    '10_minutes': '10T',
    '15_minutes': '15T',
    '30_minutes': '30T',
    
    'hour': 'H',
    'halfhourly': '30T',
    'hourly': 'H',
    '3_hours': '3H',
    '6_hours': '6H',
    '12_hours': '12H',
    
    'day': 'D',
    'daily': 'D',
    'business_day': 'B',
    
    'week': 'W',
    'weekly': 'W',
    'month': 'M',
    'monthly': 'M',
    'quarter': 'Q',
    'quarterly': 'Q',
    'year': 'Y',
    'yearly': 'Y',
    
    # Specific aliases
    '4seconds': '4S',
    '15seconds': '15S',
    '30seconds': '30S',
    '5minutes': '5T',
    '10minutes': '10T',
    '15minutes': '15T',
    '30minutes': '30T',
    '3hours': '3H',
    '6hours': '6H',
    '12hours': '12H'
}

def convert_frequency(freq):
    """
    Normalize frequency string for GluonTS
    
    Args:
        freq (str): Input frequency string
    
    Returns:
        str: Normalized GluonTS frequency
    """
    # Convert to lowercase and replace underscores
    normalized_freq = freq.lower().replace('_', '')
    
    # Look up in frequency map
    if normalized_freq in GLUONTS_FREQ_MAP:
        return GLUONTS_FREQ_MAP[normalized_freq]
    
    # If not found, return original or raise error
    return freq


from gluonts.dataset.common import ListDataset
def monash_df_to_gluonts_dataset(df, frequency="hour"):
    # Prepare the dataset
    dataset = []
    frequency = convert_frequency(frequency)
    for _, row in df.iterrows():
        # Convert string to list if it's not already a list
        if isinstance(row['series_value'], str):
            series_value = eval(row['series_value'])
        else:
            series_value = row['series_value']
        
        # Convert start timestamp
        start = pd.Timestamp(row['start_timestamp'])
        
        # Create dataset entry
        dataset.append({
            "start": start,
            "target": series_value,
            "item_id": row['series_name']
        })
    
    return ListDataset(dataset, freq=frequency)  # Adjust frequency as needed


import pandas as pd
from gluonts.dataset.common import ListDataset, TrainDatasets, MetaData
from typing import Dict, Any, Iterable

def monash_df_to_gluonts_train_datasets(df: pd.DataFrame, frequency: str = "hour", prediction_length: int = 24) -> TrainDatasets:
    """
    Converts a Monash-style DataFrame into a single GluonTS TrainDatasets object.

    Args:
        df: The input DataFrame containing time series data. Expected columns are
            'series_name', 'start_timestamp', and 'series_value'.
        frequency: The frequency of the time series data (e.g., "hour", "day").
        prediction_length: The length of the forecast horizon to be used for the test set.

    Returns:
        A TrainDatasets object containing .train and .test ListDataset objects.
    """
    
    # Map frequency string to GluonTS compatible format
    # Convert to lowercase and replace underscores
    frequency = frequency.lower().replace('_', '')
    gluonts_freq = convert_frequency(frequency)

    train_data = []
    test_data = []

    # check for timestamp series
    if 'start_timestamp' not in df.columns:
        current_timestamp = pd.Timestamp.now()
        df['start_timestamp'] = current_timestamp

    # Group by series_name to process each time series individually
    for series_name, group in df.groupby('series_name'):
        # Ensure data is sorted by timestamp
        group = group.sort_values('start_timestamp')
        
        # Extract series values and start timestamp
        series_values = [
            eval(v) if isinstance(v, str) else v
            for v in group['series_value']
        ]
        
        # Flatten the list of lists into a single list
        flat_series = [item for sublist in series_values for item in sublist]
        
        start_timestamp = pd.Timestamp(group['start_timestamp'].iloc[0])
        
        # Split the data into train and test sets
        train_target = flat_series[:-prediction_length]
        test_target = flat_series
        
        # Append to the respective lists
        train_data.append({
            "start": start_timestamp,
            "target": train_target,
            "item_id": series_name
        })
        
        test_data.append({
            "start": start_timestamp,
            "target": test_target,
            "item_id": series_name
        })

    # Create ListDatasets for train and test
    train_list_dataset = ListDataset(train_data, freq=gluonts_freq)
    test_list_dataset = ListDataset(test_data, freq=gluonts_freq)
    metadata = MetaData(
        freq=gluonts_freq,
        prediction_length=prediction_length
    )

    # Return the TrainDatasets object
    return TrainDatasets(metadata=metadata, train=train_list_dataset, test=test_list_dataset)