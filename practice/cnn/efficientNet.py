import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
import json
import os


class EFNet:
    def __init__(self, image_size, batch_size, class_list, model_name):
        self.model = EfficientNetB0(include_top=True,
                                    weights=None,
                                    input_shape=(image_size, image_size, 3),
                                    classes=len(class_list),
                                    classifier_activation='softmax')
        self.class_list = class_list
        self.image_size = image_size
        self.batch_size = batch_size
        self.save_path = 'save_model/' + model_name + '.h5'
        self.label_path = 'save_model/' + model_name + '.txt'
        self.acc_path = 'save_model/' + model_name + '.txt'

    def get_model(self):
        return self.model

    def build(self):
        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=Adam(lr=0.00005),
                           metrics=['acc'])

    def fit(self, epochs, patience, train_input, train_target, test_input, test_target):
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(self.save_path,
                                                           monitor='val_loss',
                                                           mode='min',
                                                           save_best_only=True)
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', patience=patience)

        history = self.model.fit(train_input, train_target,
                                 validation_data=(test_input, test_target),
                                 epochs=epochs,
                                 callbacks=[checkpoint_cb, early_stopping_cb])

        history_dict = history.history
        json.dump(history_dict,
                  open(self.acc_path, 'w'))
        class_list = os.listdir(self.class_list)
        _file = open(self.label_path, 'w')
        for i in class_list:
            _file.write(i + '\n')
        _file.close()


class PreProcessing:
    def __init__(self, size, data_path):
        self.img_size = size
        self.label_list = os.listdir(data_path)
        self.data_path = data_path
        self.img_cnt = 0
        for flower in self.label_list:
            self.img_cnt += len(os.listdir(data_path + flower))

    def get_train_val_set(self):
        data_input = np.ndarray(shape=(self.img_cnt, self.img_size, self.img_size, 3))
        data_target = []
        data_target = np.array(data_target)
        input_idx = 0
        label_idx = 0
        for flower in self.label_list:
            flower_img_list = os.listdir(self.data_path + flower)
            flower_label = np.array([label_idx] * len(flower_img_list))
            data_target = np.concatenate([data_target, flower_label])
            for img in flower_img_list:
                image = Image.open(self.data_path + flower + '/' + img)
                image = image.resize((self.img_size, self.img_size))
                image = np.asarray(image)
                image = image / 255.0
                data_input[input_idx] = image
                input_idx += 1
            label_idx += 1
        train_input, val_input, train_target, val_target = train_test_split(data_input,
                                                                            data_target,
                                                                            test_size=0.2,
                                                                            shuffle=True,
                                                                            stratify=data_target)
        return train_input, val_input, train_target, val_target



