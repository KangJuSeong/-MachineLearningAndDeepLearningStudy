# 분류 알고리즘

## K-최근접 이웃 분류
* 주어진 데이터에서 가장 가까운 값을 확률로 하여 각 클래스에 근접한 확률을 계산
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


fish = pd.read_csv('https://bit.ly/fish_csv')
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 이웃의 수는 3으로 함 -> 가능한 확률은 0/3, 1/3, 2/3, 3/3
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

while 1:
    idx = int(input())
    scaled_input = [test_scaled[idx]]
    target = [test_target[idx]]
    # 분류 클래스 출력
    print(kn.classes_)
    # 각각의 클래스에 근접한 확률들을 출력
    print(kn.predict_proba(scaled_input))
    print(target)
```
## 로지스틱 회귀
* 이름은 회귀지만 분류 모델.
* 선형 회귀와 동일하게 선형 방적식을 학습함.
* 로지스틱 회귀는 기본적으로 반복적인 알고리즘을 사용. `max_iter` 매개변수에서 반복 횟수를 지정하며 기본값은 100
* 준비한 데이터셋이 부족하여 반복 횟수가 부족하다는 경고가 발생하면 반복 횟수를 증가시켜 줘야 함.
* 기본적으로 `Ridge`회귀와 같이 계수의 제곱을 규제. -> L2 규제, 규제 제어 매개변수는 `C`이고 `C`는 `alpha`와 반대로 작을수록 규제가 커짐. `C`의 기본값은 1임.  
* 선형회귀와 달리 시그모이드 또는 소프트맥스 함수를 사용하여 클래스 확률을 출력
    1. 시그모이드 함수(로지스틱 함수)
        * 아주 큰 음수일 때 0이 되고, 아주 큰 양수일 때 1이 되도록 바꿔줌
        * 선형 방정식의 출력을 0과 1사이의 값으로 압축하여 이진 분류를 위해 사용
    2. 소프트맥스 함수
        * 다중 분류에서 여러 선형 방정식의 출력 결과를 정규화하여 합이 1이 되도록 함
        * 지수 함수를 사용하기 때문에 정규화된 지수 함수라고도 부름
```python
# 이진 분류
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


fish = pd.read_csv('fish.csv')
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

from sklearn.linear_model import LogisticRegression


# 불리언 인덱싱 만들어 주기 (Bream과 Smelt 일때는 True, 나머지는 False)
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
# Bream과 Smelt 데이터만 필터링 하기
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
print(lr.predict(train_bream_smelt[:5]))
print(target_bream_smelt[:5])
# 첫번째 열이 음성 클래스(0)에 대한 확률이고 두번째 열이 양성 클래스(1)에 대한 확률
print(lr.predict_proba(train_bream_smelt[:5]))
print(lr.classes_)
print(lr.coef_, lr.intercept_)
# 5개 샘플의 z값 출력
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)

# 다중 분류
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
```

## 점진적인 학습
* 앞에서 훈련한 모델을 버리지 않고 새로운 데이터에 대해서만 조금씩 더 훈련하는 방법.
### 경사 하강법
* 가장 가파른 경사를 따라 원하는 지점에 도달하는것이 목표
* 이러한 방법으로 전체 샘플을 모두 사용할 때까지 계속 경사를 조금씩 내려감
#### 1. 확률적 경사 하강법
* 훈련세트에서 무작위로 1개를 선택하여 학습
* 훈련세트에서 선택한 샘플 한개를 학습하는데 사용하고 다시 훈련세트에서 샘플을 선택, 이러한 방법을 훈련세트의 모든 샘플을 다 사용할 때까지 반복
#### 2. 미니배치 경사 하강법
* 훈련세트에서 여러개씩 선택하여 학습
* 선택된 여러개의 샘플을 통해 학습을 진행하고 훈련세트에 샘플이 다 사용될때까지 반복
#### 3. 배치 경사 하강법
* 훈련세트 전체를 선택하여 학습
* 전체 데이터를 이용하기 때문에 가장 안정적일 수 있지만 그만큼 컴퓨터 자원을 많이 사용해야 함.
* 데이터가 너무 많으면 한번에 전체 데이터를 모두 읽을 수 없는 경우가 생김
### 에포크
* 훈련세트를 한번 모두 사용하는 과정
### 손실함수
* 어떤 문제에서 머신러닝 알고리즘이 얼마나 잘못됐는지 측정하는 기준
* 상황에 따라 어떤 손실 함수를 사용할지 선택해야 함
* 샘플 하나에 대한 손실 정의
> - 로지스틱 손실 함수(이진 크로스엔트로피 손실 함수)
>     - 양성 클래스(타깃 = 1)일 때 손실은 -log(예측 확률)로 계산
>     - 음성 클래스(타깃 = 0)일 때 손실은 -log(1-예측 확률)로 계산 
### 비용함수
* 훈련 세트에 있는 모든 샘플에 대한 손실 함수의 합
```python
from sklearn.linear_model import SGDClassifier

# 확률적 경사 하강법을 제공하는 분류용 클래스
sc = SGDClassifier(loss='log',   # 손실함수를 로지스틱 회귀로 지정
                   max_iter=10,  # 에포크를 10회로 지정 
                   random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

# 1 에포크씩 이어서 훈련
sc.partial_fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
```

