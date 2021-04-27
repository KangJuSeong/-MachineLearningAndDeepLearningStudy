# 이미지를 위한 인공 신경망

## 합성곱
* 합성곱 층 => 필터(커널)을 만들고 찍어내서 만든 특성 맵
* 특성맵을 폴링층을 이용하여 크기 줄여주기
* 밀집층을 이용하여 학습
### 특성 맵
* 합성곱 층이나 풀링 층의 출력 배열을 의미. 필터 하나가 하나의 특성 맵을 만든다. 합성곱 층에서 5개 필터를 적용하면 5개 특성맵이 만들어짐.
### 패딩
* 합성곱 층의 입력 주위에 추가한 0으로 채워진 픽셀. 패딩을 사용하지 않는 것을 밸리드 패딩이라고 함. 합성곱 층의 출력 크기를 입력과 동일하게 만들기 위해 입력에 패딩을 추가하는것을 세임 패딩.
### 스트라이드
* 합성곱 층에서 필터가 입력 위를 이동하는 크기. 일반적으로 스트라이드는 1픽셀을 사용.
### 풀링
* 가중치가 없고 특성 맵의 가로세로 크기를 줄이는 역할을 수행. 대표적으로 최대 풀링과 평균 풀링이 존재하며 (2, 2)풀링이면 입력을 절반으로 줄임.
```python
from tensorflow import keras
from sklearn.model_selection import train_test_split
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

keras.utils.plot_model(model, show_shapes=True, to_file='cnn-architecture.png', dpi=100)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)

history = model.fit(train_scaled, train_target, epochs=20, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb, early_stopping_cb])
```