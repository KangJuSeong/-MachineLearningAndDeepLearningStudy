import efficientNet
import os


if __name__ == '__main__':
    model = efficientNet.EFNet(image_size=224,
                               batch_size=32,
                               class_list=os.listdir('flower_photos/'),
                               model_name='first')

    data = efficientNet.PreProcessing(size=224, data_path='flower_photos/')
    train_input, val_input, train_target, val_target = data.get_train_val_set()

    model.build()
    model.fit(epochs=100,
              patience=10,
              train_input=train_input,
              train_target=train_target,
              test_input=val_input,
              test_target=val_target)
