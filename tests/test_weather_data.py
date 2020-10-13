#!/usr/bin/env python
"""
Test the genetic lstm algorithm on a single machine over the
max planck weather dataset using a random population.
Written according to https://www.tensorflow.org/tutorials/structured_data/time_series
"""

import os
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import TimeseriesGenerator

from gentun.individuals import GeneticLSTMIndividual

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def prepare_data():
    import datetime

    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True)
    csv_path, _ = os.path.splitext(zip_path)

    df = pd.read_csv(csv_path)
    # slice [start:stop:step], starting from index 5 take every 6th record.
    df = df[5::6]

    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

    # Replace wrong measurements
    wv = df['wv (m/s)']
    bad_wv = wv == -9999.0
    wv[bad_wv] = 0.0

    max_wv = df['max. wv (m/s)']
    bad_max_wv = max_wv == -9999.0
    max_wv[bad_max_wv] = 0.0

    # edit wind direction to wind vector
    wv = df.pop('wv (m/s)')
    max_wv = df.pop('max. wv (m/s)')

    # Convert to radians.
    wd_rad = df.pop('wd (deg)') * np.pi / 180

    # Calculate the wind x and y components.
    df['Wx'] = wv * np.cos(wd_rad)
    df['Wy'] = wv * np.sin(wd_rad)

    # Calculate the max wind x and y components.
    df['max Wx'] = max_wv * np.cos(wd_rad)
    df['max Wy'] = max_wv * np.sin(wd_rad)

    # change date time to seconds
    timestamp_s = date_time.map(datetime.datetime.timestamp)

    # add periodicity by introducing sin & cos over day and year
    day = 24 * 60 * 60
    year = 365.2425 * day

    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    return df


def train_test_split(df, splits=None, normalize=True):
    if splits is None:
        splits = [0.7, 0.2, 0.1]

    n = len(df)
    train_df = df[0:int(n * splits[0])]
    val_df = df[int(n * splits[0]):int(n * (splits[0] + splits[1]))]
    test_df = df[int(n * (splits[0] + splits[1])):]

    if normalize:
        # normalize
        train_mean = train_df.mean()
        train_std = train_df.std()

        train_df = (train_df - train_mean) / train_std
        val_df = (val_df - train_mean) / train_std
        test_df = (test_df - train_mean) / train_std

    return train_df, val_df, test_df


def X_y_for_RNN(X_data, y_data, window_size, future_step=1, position=0, stride=1, return_baseline=False):
    """Builds a windowed dataset for general purpose time series forcasting/prediction.

    Args:
        X_data (2D array): The time-series to cut by windows
        y_data (1D array): the labels/values to predict
        window_size (int): The size of the window (number of timesteps in an element)
        future_step (int, optional): The delay in forecasting (must be higher than 0). Defaults to 1.
        position (int, optional): How far in the past to put the window. Defaults to 0.
        stride (int, optional): The difference between 2 following windows. Defaults to 1.
    Raises:
        ValueError: If the future_step is to small

    Returns:
        Tuple X,y: X and y (inputs and target)
    """
    if future_step < 1:
        raise ValueError("future_step must be 1 or higher")
    if position < 0:
        raise ValueError("Position must be 0 or higher")

    shift = future_step + position
    generator = TimeseriesGenerator(
        data=X_data[:len(y_data) - (shift - 1)],
        targets=y_data[shift - 1:],
        length=window_size,
        sampling_rate=1,
        stride=stride,
        batch_size=1
    )
    X, y = [], []
    for batch in generator:
        X.append(batch[0][0])
        y.append(batch[1][0])

    if return_baseline:
        baseline = y_data[window_size - 1:-future_step]
        return np.array(X), np.array(y), np.array(baseline)
    else:
        return np.array(X), np.array(y)


if __name__ == '__main__':
    from gentun import GeneticAlgorithm, Population

    data = prepare_data()
    train_df, val_df, test_df = train_test_split(data, splits=[0.7, 0.2, 0.1], normalize=True)

    predicted_feature = "T (degC)"
    window_size = 96

    X_train, y_train = X_y_for_RNN(train_df.values, train_df[predicted_feature].values, window_size=window_size, future_step=1, position=0, stride=1)
    X_val, y_val = X_y_for_RNN(val_df.values, val_df[predicted_feature].values, window_size=window_size, future_step=1, position=0, stride=1)
    X_test, y_test = X_y_for_RNN(test_df.values, test_df[predicted_feature].values, window_size=window_size, future_step=1, position=0, stride=1)

    pop = Population(
        GeneticLSTMIndividual, X_train, y_train, size=20,
        additional_parameters={'kfold': 3, 'input_shape': X_train.shape[1:]}, maximize=False
    )
    ga = GeneticAlgorithm(pop)
    ga.run(10)
