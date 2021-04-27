# 딥러닝

## 딥러닝

### 인공신경망
* 생물학적 뉴런에서 영감을 받아 만든 머신러닝 알고리즘. 이미지, 음성, 텍스트 분야에서 뛰어난 성능을 발휘하면서 주목받게 됨.
### 밀집층
* 가장 간단한 인공 신경망의 층. 인공 신경망에는 여러 종류의 층이 존재
    1. Dense
        * 신경망에서 가장 기몬 층인 밀집층을 만드는 클래스.
        * activation 매개변수에는 사용할 활성화 함수를 지정. 대표적으로 sigmoid, softmax가 있음.
        * input_shape 매개변수에는 입력되는 데이터의 크기를 넣어야 함.
        * 보통 입력 층에 매개변수가 존재.
    2. Sequential
        * 케라스에서 신경망 모델을 만드는 클래스.
        * Sequential 객체 내부에 층을 추가할 수 있고 불러 올 수 있음.
    3. compile
        * 모델 객체를 만든 후 훈련하기 전에 사용할 손실 함수와 측정 지표등을 지정하는 메소드
        * loss매개변수에는 손실함수를 지정. binary_crossentropy는 이진분류, categorical_crossentropy는 다중분류로 사용.
        * metrics 매개변수에 훈련 과정에서 측정하고 싶은 지표를 지정할 수 있음. ex)'accuracy', 'loss'
### fit
* 모델을 훈련하는 메서드
* 첫번쨰와 두번째 매개변수에 입력과 타깃 데이터를 전달.
* epochs 매개변수에 전체 데이터에 대해 반복할 에포크 횟수를 지정
### evaluate
* 모델 성능을 평가하는 메서드
* 첫번째와 두번째 매개변수에 입력과 타깃 데이터를 전달.
* compile 메서드에서 loss 매개변수에 지정한 손실 함수의 값과 metrics매개변수에서 지정한 측정 지표를 출력
```python
from tensorflow import keras
import tensorflow as tf
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

from sklearn.model_selection import train_test_split
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))  # 뉴련 개수(분류해야 할 클래스 개수), 출력에 적용 할 함수, 입력의 크기 => 밀집층 만들기(입력크기는 첫번째 입력층에 만들어야함)

model = keras.Sequential(dense)
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)
```
## 심층 신경망
* 은닉층 : 입력층과 출력층 사이에 있는 모든 층 보통 Relu 또는 sigmoid 함수를 이용
* 활성화 함수
    - 이진 분류 : 출력 층에서 sigmoid function
    - 다중 분류 : 출력 층에서 softmax function
    - 회귀 : 회귀에서는 활성화 함수 필요 없음.
    - 이미지 분류 : 출력 층에서 ReLU fuction이 효율이 좋음
```python
from tensorflow import keras
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

from sklearn.model_selection import train_test_split
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

dense0 = keras.layers.Dense(200, activation='sigmoid', input_shape=(784,), name='ConcealmentLayer1')
dense1 = keras.layers.Dense(200, activation='sigmoid', input_shape=(784,), name='ConcealmentLayer2')  # 은닉층 뉴런의 수는 정해지지 않지만 출력층의 뉴런수 보다는 많아야함(첫번째 층이므로 입력 특성 수를 넣어줘야함)
dense2 = keras.layers.Dense(10, activation='softmax', name='outputLayer')  # 출력층
model = keras.Sequential([dense0, dense1, dense2], name='ClassificationCloth')
model.summary()
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=20)
```
```python
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))  # Flatten층은 28*28 배열을 1차원으로 풀어주는 역할. 모델에 성능에는 기여 x -> kears API는 전처리를 모델 안에서 처리하려고 함
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=10)
```

## 옵티마이저
* 케라스에서 제공하는 다양한 종류의 경사하강법
###  인공신경망에서 HyperParameter
1. 은닉층의 계수 (Dense Count)
2. 뉴런 개수 (newrun Count)
3. 활성화 함수 (activation = 'sigmoid', 'softmax', 'relu')
4. 층의 종류 (layers)
5. 배치 사이즈 (batch_size default=32)
6. 에포크 (epoch)
### 옵티마이저 종류
* 기본 경사 하강법 옵티마이저
    * sgd -> 모멘텀 -> 네스테로프 모멘텀
* 적응적 학습률 옵티마이저
* RMSprop - > adam
```python
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics='accuraty')
```