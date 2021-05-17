from sklearn.preprocessing import MinMaxScaler
from craweling.parsing_data import week_parse
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.python.client import device_lib
import json


def create_model(train):
    # 모델 생성
    model = Sequential([
        # 1차원 feature map 생성
        Conv1D(filters=32, kernel_size=5,
               padding="causal",
               activation="relu",
               input_shape=[train.shape[1], train.shape[2]]),
        # LSTM
        LSTM(16, activation='tanh'),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    optimizer = Adam(0.0005)
    model.summary()
    model.compile(loss=Huber(), optimizer=optimizer, metrics=['acc'])
    return model


def normalization(code, count):
    # 최소최대 정규화를 이용하여 데이터 정규화
    scaler = MinMaxScaler()
    scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    data_scaled = scaler.fit_transform(data[scale_cols])
    data_scaled = pd.DataFrame(data_scaled, columns=scale_cols)
    # df
    return data_scaled


def split_data(input_data, label_data):
    # x_train은 학습용 데이터(Open, High, Low, Volume)
    # y_train은 학습용 데이터에 대한 Close
    # x_test는 모델에서 예측된 값과 비교 할 실제값(Open, High, Low, Volume)
    # y_test는 실제값에 대한 Close
    input_train, input_test, label_train, label_test = train_test_split(input_data,
                                                                        label_data,
                                                                        test_size=0.2,
                                                                        random_state=0,
                                                                        shuffle=True)
    input_val, input_test, label_val, label_test = train_test_split(input_test,
                                                                    label_test,
                                                                    test_size=0.4,
                                                                    random_state=0,
                                                                    shuffle=True)
    return input_train, label_train, input_val, label_val, input_test, label_test


def windowed_dataset(x_data, y_data, window_size):
    dataX = []
    dataY = []
    for i in range(len(x_data) - window_size):
        dataX.append(np.array(x_data.iloc[i:i+window_size]))
        dataY.append(np.array(y_data.iloc[i+window_size]))
    return np.array(dataX), np.array(dataY)


def test_windowed_dataset(x_data):
    dataX = []
    dataX.append(np.array(x_data.iloc[0:20]))
    return np.array(dataX)


def train(train_x, train_y, val_x, val_y, test_x, test_y):
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    device_lib.list_local_devices()
    tf.config.list_physical_devices('GPU')
    model = create_model(train_x)
    history = model.fit(train_x, train_y,
                        batch_size=32,
                        validation_data=(val_x, val_y),
                        epochs=200)
    model.save('stock_price.h5')
    history_dict = history.history
    json.dump(history_dict, open('history.txt', 'w'))
    results = model.evaluate(test_x, test_y, batch_size=32)
    print(results)


def test(test_data):
    model = create_model(test_data)
    model.load_weights('stock_price.h5')
    pred = model.predict(test_data)
    return pred