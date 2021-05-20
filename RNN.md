# 텍스트를 위한 인공 신경망

## 순차 데이터와 순환 신경망

### 1. 순차 데이터
* 순차 데이터는 텍스트나 시계열 데이터와 같이 순서에 의미가 있는 데이터를 말함.
* 예시로 단어로 이루어진 문장이나 일별 온도를 기록한 데이터와 같이 순서가 데이터에 중요한 요소일 수 있음.

### 2. 순환 신경망
* 기존에 했던 CNN 은 이전에 사용한 샘플의 계산을 수행하고 나면 해당 샘플은 버려지는데 이를 피드포워드 신경망이라 부르며, 데이터의 흐름이 앞으로만 전달됨.
* 순환 신경망에서는 이전에 처리했던 샘플을 다음 샘플을 처리하는데 재사용할 수 있음.
* 순환 신경망은 일반적인 완전 연결 신경망과 거의 비슷, 하지만 여기서 이전 데이터의 처리 흐름을 순환하는 고리 하나를 추가해줘야 함.
> ### 타임스텝
>   * 하나의 뉴런에 A를 입력했을 때 출력은 Oa
>   * 같은 뉴런에 B를 입력할 때 Oa도 함께 입력하여 나온 출력이 Ob
>   * 같은 뉴런에 C를 입력할 때 Ob도 함께 입력하여 나온 출력이 Oc
>   * Oc 출력에는 A, B, C 모든 입력에 대한 정보가 담겨있음.
>   * 타임스텝이 오래될수록 순환되는 정보는 희미해짐.
> ### 셀
>   * 순환 신경망에서는 특별히 층을 셀이라고 부름.
>   * 한 셀에는 여러 개의 뉴런이 있지만 완전 연결 신경망과 달리 뉴런을 모두 표시하지 않고 하나의 셀로 층을 표시.
> ### 은닉 상태
>   * 셀의 출력을 은닉 상태라고 부름.
>   * 은닉층의 활성화 함수로는 하이퍼볼릭 탄젠트 함수(tanh^2)을 주로 사용.
>   * tanh 함수도 S자 모양을 띠기 때문에 종종 시그모이드 함수라고 부르기도 하지만 시그모이드 함수와 달리 -1 ~ 1 사이의 범위를 가짐.

### 3. 셀의 가중치와 입출력
* 순환 신경망에서 각 셀에 입력된 값은 가중치 Wb와 곱해진다고 가정할 때 모든 셀에서 사용되는 가중치는 같음.
* 가장 첫번째 타임스텝에서 사용되는 이전 은닉 상태는 맨 처음 샘플을 입력할 때는 이전 타임스텝이 없으므로 간단히 모두 0으로 초기화함.
> ### 셀의 가중치 크기 계산
>   1. 입력층에 x1 ~ x4 특성 4개가 존재하고, 순환층에는 r1 ~ r3까지 3개의 뉴런이 있다고 가정.
>   2. 가중치 Wx의 크기는 4 x 3 = 12 로 12개.  
> ### 은닉 상태를 위한 가중치 크기 계산
>   1. r1의 은닉 상태가 다음 타임스텝에 재사용될 때 r1과 r2, r3 모두에게 전달.
>   2. r2의 은닉 상태도 마찬가지로 r1, r2, r3 모두에게 전달.
>   3. r3의 은닉 상태도 같음.
>   4. 이 순환층에서 은닉상태를 위한 가중치 Wb는 3 x 3 = 9 로 9개.
> ### 모델 파라미터 수
>   1. 가중치에 절편을 더하기.
>   2. 예시에는 뉴런마다 하나의 절편이 존재.
>   3. 12 + 9 + 3 = 24 로 24개의 모델 파라미터 존재.
> ### 순환층의 입력
>   * 일반적으로 샘플마다 2개의 차원을 가지며, 하나의 샘플을 하나의 시퀀스라고 말함.
>   * 시퀀스 안에는 여러 개의 아이템이 들어 있으며, 여기에서 시퀀스의 길이가 타임스텝 길이가 됨.
>       > ex) I am a boy => 1개의 샘플안에 4개의 단어가 있고 각 단어는 3개로 표현될 수 있음. (1, 4, 3)
>   * 이런 입력이 순환층을 통과하면 두 번째, 세 번째 차원이 사라지고 순환층의 뉴련 개수만큼 출력됨.
> ### 순환층의 출력
>   * 결론적으로 셀이 모든 타임스텝에서 출력을 만들지만 최종 출력은 마자믹 타임스텝의 은닉 상태만 출력으로 내보냄.
>   * 마치 임력된 시퀀스 길이를 모두 읽어서 정보를 마지막 은닉 상태에 압축하여 전달하는 것처럼 볼 수 있음.
>   * 정보를 기억하는 메모리를 가진다고 표현할 수 있음.
>   * 다중 분류일 경우 출력층에 클래스 개수만큼 뉴런을 두고 활성화 함수를 소프트맥스 사용.
>   * 이진 분류일 경우 출력층에 하나의 뉴런을 두고 활성화 함수를 시그모이드 사용.
>   * 마지막 출력 시 셀의 출력이 1차원이므로 Flatten 클래스로 펼쳐줄 필요가 없고 출력 그대로 밀집층에 사용할 수 있음.

## 순환 신경망으로 IMDB 리뷰 분류하기
* 텍스트 데이터의 경우 단어를 숫자로 바꿔줘야 하며 일반적인 방법으로는 데이터에 등장하는 단어마다 고유한 정수를 부여함.
* 위와 같은 방식으로 분리된 단어들을 토큰이라고 부름.
```python
# 데이터 준비
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)
train_input, val_input, train_target, val_target = train_test_split(train_input, train_target,
                                                                    test_size=0.2,
                                                                    random_state=42)

# pad_sequences는 maxlen 보다 긴 시퀀스는 앞에서부터 토큰을 자름 -> 일반적으로 시퀀스의 뒷부분의 정보가 더 유용하리라 기대하기 때문
# 뒤에서부터 자르고 싶다면 truncating='post' 로 변경
# padding='post' 로 변경하면 패딩(0)을 뒤에서 부터 삽입
train_seq = pad_sequences(train_input, maxlen=100, truncating='pre', padding='pre')
val_seq = pad_sequences(val_input, maxlen=100, truncating='pre', padding='pre')
```

### 1. 순환 신경망 만들기
```python
from tensorflow import keras

model = keras.Sequential()
# input 이 (100, 500) 인 이유는 한 시퀀스의 최대 토큰 수는 100개이며 총 500개의 단어가 존재하고 해당 단어들을 원-핫 인코딩 한 배열의 크기는 500이다
# SimpleRNN 에서 기본 활성화 함수는 tanh
model.add(keras.layers.SimpleRNN(8, input_shape=(100, 500)))
# 출력층, 이진 분류 이므로 뉴런은 1개, 활성화 함수는 sigmoid
model.add(keras.layers.Dense(1, activation='sigmoid'))
# 입력 데이터의 토큰 값을 원핫인코딩으로 바꾸기
train_oh = keras.utils.to_categorical(train_seq)
val_oh = keras.utils.to_categorical(val_seq)
```

### 2. 순환 신경망 훈련하기
```python
import matplotlib.pyplot as plt

rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['acc'])
cp = keras.callbacks.ModelCheckpoint('best-simplernn-model.h5', save_best_only=True)
es = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model.fit(train_oh, train_target, epochs=100, batch_size=64,
                    validation_data=(val_oh, val_target),
                    callbacks=[cp, es])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```

### 3. 단어 임베딩을 사용하기
* 순환 신경망에서 텍스트를 처리할 때 즐겨 사용하는 방법.
* 각 단어를 고정된 크기의 실수 벡터로 바꾸어 주고 이런 단어 임베딩으로 만들어진 벡터는 원-핫 인코딩된 벡터보다 훨씬 의미 있는 값으로 채우져 있기 때문에 자연어 처리에서 더 좋은 성능을 내는 경우가 많음.
* 케라스에서 `keras.layers.Embedding` 을 이용하여 임베딩 벡터를 만들어 주는 층 생성.
```python
model2 = keras.Sequential()
# 첫번째 매개변수는 어휘 사전의 크기, 두번째 매개변수는 임베딩 벡터의 크기, 세번째 매개변수는 입력 시퀀스의 길이
model2.add(keras.layers.Embedding(500, 16, input_length=100))
model2.add(keras.layers.SimpleRNN(8))
model2.add(keras.layers.Dense(1, activation='sigmoid'))

rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['acc'])
cp = keras.callbacks.ModelCheckpoint('best-simplernn-model.h5', save_best_only=True)
es = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model.fit(train_oh, train_target, epochs=100, batch_size=64,
                    validation_data=(val_oh, val_target),
                    callbacks=[cp, es])

# 이전 모델과 비슷한 성능을 냈지만 순환층의 가중치 개수와 훈련 세트 크기가 훨씬 줄었음
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```

## LSTM과 GRU 셀

### 1. LSTM 구조
* Long Shor-Term Memory 의 약자로 단기 기억을 오래 기억하기 위해 고안된 구조.
* LSTM 에는 입력과 가중치를 곱하고 절편을 더해 활성화 함수를 통과시키는 구조를 여러개 가지고 있음.
* 은닉 상태는 입력과 이전 타임스텝의 은닉 상태를 가중치에 곱한 후 활성화 함수를 통과시켜 다음 은닉 상태를 만듬.
* 기본 순환층과는 달리 시그모이드 할성화 함수를 사용하며 tnnh 활성화 함수를 통과한 어떤 값가 곱해져서 은닉 상태를 만듬.
* LSTM 에는 순환되는 상태과 은닉 상태와 셀 상태 두가지이며 셀 상태는 다음 층으로 전달되지 않고 LSTM 셀에서 순환만 되는 값.
    > ### 셀 상테 계산
    > 1. 셀 상태를 c, 은닉 상태를 h로 함.
    > 2. 입력과 은닉 상태를 또 다른 가중치 Wf에 곱한 다음 시그모이드 함수를 통과.
    > 3. 이전 타임스텝의 셀 상태와 곱하여 새로운 셀 상태를 만듬.
    > 4. 만들어진 셀 상태가 tanh 함수를 통과하여 새로운 은닉 상태를 만드는데 기여.
    > 5. 추가적으로 2개의 작은 셀이 더 추가되어 셀 상태를 만드는데 기여하는데 이전과 마찬가지로 입력과 은닉 상태를 각기 다른 가중치에 곱한 다음, 하나는 시그모이드 함수를 통과시키고 다른 하나는 tanh 함수를 통과시킴.
    > 6. 두 결과를 곱한 후 이전 셀 상태와 더하면 최종적인 다음 셀 상태가 생김. 

### 2. LSTM 신경망 훈련
```python
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)
train_input, val_input, train_target, val_target = train_test_split(train_input, train_target,
                                                                    test_size=0.2,
                                                                    random_state=42)

train_seq = pad_sequences(train_input, maxlen=100)
vla_seq = pad_sequences(val_input, maxlen=100)

model = keras.Sequential()
model.add(keras.layers.Embedding(500, 16, input_length=100))
model.add(keras.layers.LSTM(8))
model.add(keras.layers.Dense(1, activation='sigmoid'))

rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['acc'])
cp = keras.callbacks.ModelCheckpoint('best-simplernn-model.h5', save_best_only=True)
es = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model.fit(train_oh, train_target, epochs=100, batch_size=64,
                    validation_data=(val_oh, val_target),
                    callbacks=[cp, es])

# 결과에서 기본 순환층보다 LSTM 이 과대적합을 억제하면서 훈련을 더 잘 수행한 것으로 보임
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```

### 3. 순환층에 드롭아웃 적용
* 순환층은 자체적으로 드롭아웃 기능을 제공.
* `SimpleRNN` 과 `LSTM` 클래스 모두 `dropout` 매개변수와 `recurrent_dropout` 매개변수를 가지고 있음.
* `dropout` 매개변수는 셀의 입력에 드롭아웃을 적용하고 `recurrent_dropout`은 순환되는 은닉 상태에 드롭아웃을 적용.
```python
# 30%의 입력을 드롭아웃하여 이전 모델보다 훈련 손실과 검증 손실간의 차이가 더 좁혀진것을 확인
model.add(keras.layers.LSTM(8, dropout=0.3))
```

### 4. 2개의 층 연결
* 순환층의 은닉 상태는 샘플의 마지막 타임스텝에 대한 은닉 상태만 다음 층으로 전달하며 순환층을 쌓게 되면 모든 순환층에 순차 데이터가 필요.
* 앞쪽의 순환층이 모든 타임스텝에 대한 은닉 상태를 출력해 주어야 함.
* 모든 타임스텝의 은닉 상태를 출력하기 위해서는 `return_sequences` 매개변수를 True로 지정하면 됨.
```python
# 2개의 LSTM 층을 이용하여 1개의 LSTM을 이용한 모델보다 과대적합을 더 잘 제어하면서 손실을 최대한 낮출수 있었음
model = keras.Sequential()
model.add(keras.layers.Embedding(500, 16, input_length=100))
# 2개의 LSTM 층을 쌓았고 모든 드롭아웃을 0.3으로 지정, 첫번째 LSTM 층에서는 모든 타임스텝의 은닉 상태를 출력 output => (None, 100, 8)
model.add(keras.layers.LSTM(8, dropout=0.3, return_sequences=True))
# 두번쨰 층은 마지막 타임스텝의 은닉 상태만 출력 output => (None, 8)
model.add(keras.layers.LSTM(8, dropout=0.3))
model.add(keras.layers.Dense(1, activation='sigmoid'))
```

### 5. GRU 구조
* GRU 는 Gated Recurrent Unit의 약자로 LSTM 을 간소화한 버전으로 생각할 수 있음.
* GRU 셀에는 은닉 상태와 입력에 가중치를 곱하고 절편을 더하는 작은 셀이 3개 들어있음.
* Wz를 사용하는 셀의 출력이 은닉 상태에 바로 곱해져 삭제 게이트 역할, 이와 똑같은 출력을 1에서 뺀 다음 Wg를 사용하는 셀의 출력에 곱하여 입력되는 정보를 제어하는 역할, Wr을 사용하는 셀에서 출력된 값은 Wg 셀이 사용할 은닉 상태의 정보를 제어.  
* 2개는 시그모이드 활성화 함수를 사용하고 하나는 tanh 활성화 함수를 사용.
* GRU 셀은 LSTM 보다 가중치가 적기 때문에 계산량이 적지만 LSTM 못지않은 좋은 성능을 내는 것으로 알려짐.

### 6. GRU 신경망 훈련
```python
model = keras.Sequential()
model.add(keras.layers.Embedding(500, 16, input_length=100))
# GRU 셀 추가
model.add(keras.layers.GRU(8))
model.add(keras.layers.Dense(1, activation='sigmoid'))
```