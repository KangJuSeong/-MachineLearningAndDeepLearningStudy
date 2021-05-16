# 딥러닝

## 인공신경망
* 생물학적 뉴런에서 영감을 받아 만든 머신러닝 알고리즘. 이미지, 음성, 텍스트 분야에서 뛰어난 성능을 발휘하면서 주목받게 됨.
1. 출력층
    * 각각의 클래스에 대한 z값을 계산하고 이를 바탕으로 클래스를 예측하여 최종값을 만드는 층.
2. 뉴런
    * 인공 신경망에서는 z 값을 계산하는 단위를 뉴런이라고 함.
    * 뉴련에서는 선형 계산이 전부.
3. 입력층
    * 어떠한 입력되는 특성들의 입력을 받는 층.
    * 특별한 계산을 수행하지는 않음.
4. 밀집층(완전 연결층)
    * 가장 간단한 인공 신경망의 층.
    * 하나의 z 값들을 계산하여 예측에 필요한 새로운 z 값들을 만들어 냄.   
    * `dense = keras.layers.Dense(10, activation='softmax', input_shipe(784))`
    > Dense
    > * 신경망에서 가장 기본 층인 밀집층을 만드는 클래스.
    > * input_shape 매개변수에는 입력되는 데이터의 크기를 넣어야 함.
    > * 보통 입력 층에 매개변수가 존재.
    > * activation 매개변수에는 사용할 활성화 함수를 지정. 대표적으로 sigmoid, softmax가 있음.
    >  > 활성화 함수 
    >  > * 뉴련의 선형 방정식 계산 결과에 적용되는 함수
    >  >    1. softmax
    >  >        - 여러개의 뉴런에서 출력되는 값을 확률로 바꾸기 위해서 사용.  
    >  >    2. sigmoid
    >  >        - 2개의 클래스를 분류하는 이진 분류에서 사용.
        
### Sequential
`model = keras.Sequential(dense)`
* 케라스에서 신경망 모델을 만드는 클래스.
* Sequential 객체 내부에 층을 추가할 수 있고 불러 올 수 있음.

### compile
`model.compile(loss='sparse_categorical_crossentoropy', metrics='accuracy')`
* 모델 객체를 만든 후 훈련하기 전에 사용할 손실 함수와 측정 지표등을 지정하는 메소드
* loss 매개변수에는 손실함수를 지정. 
* sparse 가 붙으면 정수로된 타겟값을 사용해 크로스 엔트로피 손실을 계산하겠다는 뜻.
* sparse 가 붙지 않는다면 타겟값을 원-핫 인코딩으로 준비해야 함.
  > 1. binary_crossentropy -> 이진분류   
  > 2. categorical_crossentropy -> 다중분류
* metrics 매개변수에 훈련 과정에서 측정하고 싶은 지표를 지정할 수 있음. ex)'accuracy', 'loss'

### fit
`model.fit(train_input, train_target, epochs=5)`
* 모델을 훈련하는 메서드
* 첫번쨰와 두번째 매개변수에 입력과 타겟 데이터를 전달.
* epochs 매개변수에 전체 데이터에 대해 반복할 에포크 횟수를 지정

### evaluate
`model.evaluate(test_input, test_target)`
* 모델 성능을 평가하는 메서드
* 첫번째와 두번째 매개변수에 학습에 이용되지 않은 입력과 타겟 데이터를 전달.
* compile 메서드에서 loss 매개변수에 지정한 손실 함수의 값과 metrics 매개변수에서 지정한 측정 지표를 출력
```python
from tensorflow import keras
import tensorflow as tf
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

from sklearn.model_selection import train_test_split
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
train_scaled, val_scaled, train_target, val_target = tr ain_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
# 뉴련 개수(분류해야 할 클래스 개수), 출력에 적용 할 함수, 입력의 크기 => 밀집층 만들기(입력크기는 첫번째 입력층에 만들어야함)
dense = keras.layers.Dense(10, activation='softmax', input_shape=(train_scalde.shape[0],))
model = keras.Sequential(dense)
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)
```
## 심층 신경망
1. 은닉층
    * 입력층과 출력층 사이에 있는 모든 층 보통 Relu 또는 sigmoid 함수를 이용
    * 출력층에서 사용되는 활성화 함수는 제한적이지만 은닉측의 활성화 함수는 다양하게 사용 가능.  
    > * 활성화 함수
    >     - 이진 분류 : 출력 층에서 sigmoid 함수.
    >     - 다중 분류 : 출력 층에서 softmax 함수.
    >     - 회귀 : 회귀에서는 활성화 함수 필요 없음.
    >     - 이미지 분류 : 출력 층에서 ReLU 함수가 효율적.
    >         > * Relu
    >         >   1. 층이 많은 심층 신경망일수록 그 효과가 누적되어 학습을 더 어렵게 만드는데 이를 개선하기 위해 제안된 함수.
    >         >   2. z가 0보다 크면 z를 출력하고 z가 0보다 작으면 0을 출력하는 방식
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
# summary() 메서드를 호출하면 층에 대한 유용한 정보를 얻을 수 있음.
model.summary()
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=20)
```
2. Flatten층
    * 배치 차원을 제외하고 나머지 입력 차원을 모두 일렬로 펼치는 역할.
    * 모델에 성능에는 기여하지는 않음.
```python
model = keras.Sequential()
 # Flatten층은 28*28 배열을 1차원으로 풀어주는 역할.
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=10)
```

### HyperParameter
1. 은닉층의 계수 (Dense Count)
2. 뉴런 개수 (newrun Count)
3. 활성화 함수 (activation = 'sigmoid', 'softmax', 'relu')
4. 층의 종류 (layers)
5. 배치 사이즈 (batch_size default=32)
6. 에포크 (epoch)
7. 기본 경사 하강법(optimizer)
8. RMSporp의 학습률 

> 옵티마이저  
> 1. 기본적 경사 하강법 옵티마이저
>   * 케라스에서 다양한 종류의 경사 하강법 알고리즘을 제공하는데 이것을 옵티마이저라고 부름.
>   * 기본 옵티마이저는 SGD (확률적 경사 하강법)
>       ```python
>       model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics='accuraty')
>       ```
>   * SGD 클래스의 학습률 기본값이 0.01일 때 이를 바꾸고 싶다면 `learning_rate` 매개변수에 지정하여 사용.
>   * `momentum` 매개변수를 지정하여 이전의 그레이디언트를 가속도처럼 사용하는 모멘텀 최적화를 사용. 보통은 0.9 이상을 적용
>   * `nesterov` 매개변수를 Ture 로 바꾸면 네스테로프 모멘텀 최적화를 사용(네스테로프 가속 경사)
>       ```python
>       # 학습률 조정
>       sgd = keras.optimizers.SGD(learning_rate=0.1)
>       # 네스테로프 모멘텀 최적화
>       nestrov_momentum_sgd = keras.optimizers.SGD(momentum=0.9, nesterov=True) 
>       ```
> 2. 적응적 학습률 옵티마이저
>   * 모델이 최적점에 가까이 갈수록 학습률을 낮출수 있다.
>   * 안정적으로 최적점에 수렴할 가능성이 높은데 이런 학습률을 적응적 학습률이라고 함.
>   * 이런 방식들은 학습률 매개변수를 튜닝하는 수고를 덜 수 있는 것이 장점.
>   * 대표적으로 `Adagrad` 와 `RMSprop` 그리고 모멘텀 최적화와 `RMSprop`의 장점을 접목한 `Adam`이 있음.
>   * 기본적으로 `learnign_rate` 매개변수의 값은 0.001 을 사용.
>       > 1. Adagrad    
>           `adagrad = keras.optimizers.Adagrad()`
>       > 2. RMSprop    
>           `rmsprop = keras.optimizers.RMSprop()`
>       > 3. Adam   
>           `adam = keras.optimizers.Adam()` 

## 신경망 모델 훈련

### 손실 곡선