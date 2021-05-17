import crawling
from train import normalization, split_data, train
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


if __name__ == '__main__':
    count = 8100
    data = week_parse('005930', str(count+1))
    data = pd.DataFrame(data)
    input_data = np.ndarray(shape=(count-19, 20, 5))
    label_data = list()
    scaler = MinMaxScaler()
    scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    norm_data = scaler.fit_transform(data[scale_cols])
    for i in range(count-19):
        input_data[i] = norm_data[i:i+20]
        label_data.append(norm_data[i+20:i+21][0][3])
    label_data = np.asarray(label_data)

    input_train, label_train, input_val, label_val, input_test, label_test = split_data(input_data=input_data,
                                                                                        label_data=label_data)
    train(input_train, label_train, input_val, label_val, input_test, label_test)
