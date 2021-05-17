# 이미지를 위한 인공 신경망

## 합성곱 신경망의 구성 요소

### 합성곱
`keras.layers.Conv2D(10, kernel_size(3, 3), activation='relu, padding='same', strides=2)`
* 합성곱 층 => 필터(커널)을 만들고 찍어내서 만든 특성 맵.
* 최종적으로 나온 특성맵들을 모두 펼쳐 밀집층의 입력으로 사용됨.
* 필터(커널)의 사이즈는 하이퍼파라미터.
* 첫번째 매개변수는 필터의 개수.

### 특성 맵
* 합성곱 층이나 풀링 층의 출력 배열을 의미. 필터 하나가 하나의 특성 맵을 만듬.
* 4x4 의 2차원 배열에서 3x3 필터를 이용하면 4개의 합성곱이 출력되고 해당 합성곱 출력들로 만들어진 2x2의 출력을 특성맵이라고 함.

### 패딩
`padding='same'`
* 합성곱 층의 입력 주위에 추가한 0으로 채워진 픽셀.
* 패딩을 사용하지 않는 것을 밸리드 패딩. 
* 합성곱 층의 출력 크기를 입력과 동일하게 만들기 위해 입력에 패딩을 추가하는것을 세임 패딩.
* 4x4 크기의 입력에 3x3 크기의 커널을 이용하여 4x4 크기의 출력을 만들기 위해서는 패딩을 통해 입력이 6x6 인것 처럼 크기를 늘려 줌.
* 패딩의 역할은 순전히 커널이 찍어내는 횟수를 늘려주는 것 밖에 없음.
* 만약 입력에서 중요한 정보가 모서리에 있다면 특성 맵으로 잘 전달되지 않을 수 있으므로 패딩을 통해 모서리에 있는 정보까지 특성맵에 잘 반영되도록 해줄 수 있음.

### 스트라이드
`strides=1` or `strides=(1, 1)`
* 합성곱 층에서 필터가 입력 위를 이동하는 크기. 
* 일반적으로 스트라이드는 1픽셀을 사용.
* 튜플을 이용하여 오른쪽으로 이동하는 크기와 아래쪽으로 이동하는 크기를 지정 가능.
* 스트라이드가 증가하면 커널이 찍어내는 합성곱 층의 출력이 줄어듬.
* 대부분 1보다 큰 스트라이드를 사용하는 경우는 없고, 기본값을 그대로 사용.

### 풀링
`keras.layers.MaxPooling2D(2)`  
`keras.layers.AveragePooling2D(2)`
* 가중치가 없고 특성 맵의 가로세로 크기를 줄이는 역할을 수행하며 특성맵의 개수가 줄어들진 않음.
* 대표적으로 풀링으로 찍은 영역에서 가장 큰 값을 고르는 최대 풀링과 평균값을 고르는 평균 풀링이 존재.
* (2,2,3) 크기의 특성 맵에 (2, 2) 크기의 풀링을 이용하면 (1,1,3) 크기의 특성맵으로 바꿀 수 있음.
* 매개변수의 첫번째는 2로 입력하면 (2, 2) 크기이고 스트라이드는 2로 절반을 줄이겠다는 뜻.

```python
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

model = keras.Sequential()

# 두개의 합성곱층과 풀링층 -> 최종 특성맵의 shape 는 (7,7,64)
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2))

# 특성 맵을 일렬로 펼치고 10개 뉴런을 가진 출력층에서 확률을 계산하기
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
# 드롭아웃 층이 은닉층의 과대적합을 막아 성능을 조금더 개선해줄 수 있음
model.add(keras.layers.Dropout(0.4))
# 10개의 클래스를 분류하므로 출력층에서는 10개의 뉴런, 소프트맥스 활성화 함수를 사용
model.add(keras.layers.Dense(10, activation='softmax'))

# 모델의 층들을 그래프로 표현
keras.utils.plot_model(model, show_shapes=True, to_file='cnn-architecture.png', dpi=300)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)

history = model.fit(train_scaled, train_target, epochs=20, validation_data=(val_scaled, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```

## 합성곱 신경망의 시각화

### 가중치 시각화
* 일반적으로 절편은 시각적으로 의미가 있지는 않음.
* 모델이 어떤 가중치를 학습했는지 확인할 수는 있음.
```python
from tensorflow import keras


model = keras.models.load_model('best-cnn-model.h5')
conv = model.layers[0]
# 첫번째는 가중치의 크기, 두번째는 절편의 개수
print(conv.weights[0].shape, conv.weights[1].shape)

# 가중치의 평균과 표준편차 계산
conv_weights = conv.weights[0].numpy()
print(conv_weights.mean(), conv_weights.std())

# 32개의 커널을 컬러맵으로 표현
fig, axs = plt.subplots(2, 16)
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(conv_weights[:, :, 0, i*16+j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')
plt.show()

# 훈련이 되어있지 않은 모델
no_training_model = keras.Sequential()
no_training_model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28, 28, 1)))

# 첫번째 합성곱 층의 shape 와 가중치의 평균과 표준편차 출력
no_training_conv = no_training_model.layers[0]
print(no_training_conv.weights[0].shape)
no_training_weights = no_training_conv.weights[0].numpy()
print(no_training_weights.mean(), no_training_weights.std())

# 32개의 커널을 컬러맵으로 출력
fig, axs = plt.subplots(2, 16)
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(no_training_weights[:, :, 0, i*16+j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')
plt.show()
```

### 함수형 API
* 입력이 2개거나 출력이 2개일 수도 있는 경우에는 Sequential 클래스를 사용하기 어려우므로 함수형 API 를 사용.
* 함수형 API 를 통해 모델을 만들면 중간에 다양한 형태로 층을 연결할 수 있음.
* 각각의 층에서 나온 출력값들을 저장하고 사용하기 때문에 특성 맵 시각화를 하는데 편리함.
```python
dense1 = keras.layers.Dense(100, activation='sigmoid')
dense2 = keras.layers.Dense(10, activation='softmax')

inputs = keras.Input(shape=(784,))
hidden = dense1(inputs)
outputs = dense2(hidden)
model = keras.models.Model(inputs, outputs)

# 모델의 입력을 간단하게 알 수 있음.
print(model.input)

# 이런 방식으로 model.input과 model.layers[0].output 을 연결하는 새로운 conv_acti 모델을 만들 수 있음
conv_acti = keras.models.Model(model.input, model.layers[0].output)
```

### 특성맵 시각화
* 함수형 API 를 통해서 원하는 위치의 합성곱 층에서 나온 출력을 통해 특성맵 그려서 시각화할 수 있음.
```python
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

load_model = keras.models.load_model('best-cnn-model.h5')

dense1 = keras.layers.Dense(100, activation='sigmoid')
dense2 = keras.layers.Dense(10, activation='softmax')

# 합성곱 층 한개에서 나온 출력을 수치화 하는 함수형 모델 
inputs = keras.Input(shape=(784,))
hidden = dense1(inputs)
outputs = dense2(hidden)
model = keras.models.Model(inputs, outputs)
# 첫번째 합성곱 층 연결
conv_act = keras.models.Model(load_model.input, load_model.layers[0].output)

# 테스트용 이미지를 첫번째 합성곱 층의 입력으로 넣고 출력으로 나온 특성 맵 확인하기 
input_data1 = train_input[0:1].reshape(-1, 28, 28, 1) / 255.0
feature_maps = conv_act.predict(input_data1)
print(feature_maps.shape)

# 특성맵 맷플롯립으로 시각화 하기 
fig, axs = plt.subplots(4, 8)
for i in range(4):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0, :, :, i*8+j])
        axs[i, j].axis('off')
plt.show()

# 두번째 합성곱 층 연결
conv_act2 = keras.models.Model(load_model.input, load_model.layers[2].output)

input_data2 = train_input[0:1].reshape(-1, 28, 28, 1) / 255.0
feature_maps2 = conv_act2.predict(input_data2)
print(feature_maps2.shape)

# 두번째 합성곱 층에서 나온 출력 특성 맵 시각화
fig, axs = plt.subplots(8, 8)
for i in range(8):
    for j in range(8):
        axs[i, j].imshow(feature_maps2[0, :, :, i*8+j])
        axs[i, j].axis('off')
plt.show()
```