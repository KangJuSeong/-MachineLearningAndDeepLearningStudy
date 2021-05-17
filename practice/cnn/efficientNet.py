import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0


IMAGE_SIZE = 224

flower_name = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
flower_path = ['/flower_photos/daisy/',
               '/flower_photos/dandelion/',
               '/flower_photos/roses/',
               '/flower_photos/sunflowers/',
               '/flower_photos/tulips/']

daisy = os.listdir(flower_path[0])
dandelion = os.listdir(flower_path[1])
roses = os.listdir(flower_path[2])
sunflowers = os.listdir(flower_path[3])
tulips = os.listdir(flower_path[4])

data_input = np.ndarray(shape=(len(daisy)+len(dandelion)+len(roses)+len(sunflowers)+len(tulips), IMAGE_SIZE, IMAGE_SIZE, 3),
                        dtype=np.float32)
data_target = np.array([0]*len(daisy) + [1]*len(dandelion) + [2]*len(roses) + [3]*len(sunflowers) + [4]*len(tulips))
flower_img = [daisy, dandelion, roses, sunflowers, tulips]
array_idx = 0
flower_idx = 0
for flower in flower_img:
    for img in flower:
        image = Image.open(flower_path[flower_idx] + img)
        image = image.resize(image_size)
        image = np.asarray(image)
        image = image / 255.0
        data_input[array_idx] = image
        array_idx += 1
    flower_idx += 1

train_input, test_input, train_target, test_target = train_test_split(data_input,
                                                                      data_target,
                                                                      test_size=0.2,
                                                                      shuffle=True,
                                                                      stratify=data_target,
                                                                      random_state=32)

model = EfficientNetB0(include_top=True,
                       weights=None,
                       input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                       classes=len(flower_name),
                       classifier_activation='softmax')
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=0.00005),
              metrics=['acc'])

epochs = 150

checkpoint_cb = keras.callbacks.ModelCheckpoint('flower_classification.h5',
                                                monitor='val_loss',
                                                mode='min',
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=20)

history = model.fit(
  train_input,
  train_target,
  validation_data=(test_input, test_target),
  epochs=epochs,
  callbacks=[checkpoint_cb, early_stopping_cb]
)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

image = Image.open('daisy.jpg')
image = image.resize((224, 224))
array = np.asarray(image) / 255.0
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
data[0] = array
plt.imshow(data[0])
plt.show()

label = model.predict(data)
result = np.argmax(label[0])
if result == 0:
    print('daisy')
elif result == 1:
    print('dandelion')
elif result == 2:
    print('roses')
elif result == 3:
    print('sunflowers')
elif result == 4:
    print('tulips')
