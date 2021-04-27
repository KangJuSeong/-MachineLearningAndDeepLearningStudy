# 분류 알고리즘

## 로지스틱 회귀와 다중분류
* 선형 방정식을 사용한 분류 알고리즘.
* 선형회귀와 달리 시그모이드 또는 소프트맥스 함수를 사용하여 클래스 확률을 출력
    1. 시그모이드 함수
        * 선형 방정식의 출력을 0과 1사이의 값으로 압축하여 이진 분류를 위해 사용
    2. 소프트맥스 함수
        * 다중 분류에서 여러 선형 방정식의 출력 결과를 정규화하여 합이 1이 되도록 함
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, softmax


fish = pd.read_csv('https://bit.ly/fish_csv')

fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

kn = KNeighborsClassifier(n_neighbors=3) # 최근접한 클래스 개수에 대한 확률값 계산 (이웃수는 3개)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

print(kn.predict(test_scaled[:5]))

proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))

print( "순서 : ", kn.classes_)
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]
lr = LogisticRegression() # 이진 분류
lr.fit(train_bream_smelt, target_bream_smelt)
print(lr.predict(train_bream_smelt[:5]))
print(lr.predict_proba(train_bream_smelt[:5])) # 음성클래스와 양성클래스에 대한 확률
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions) # 양성클래스에 대한 z값
print(expit(decisions)) # decisions 값에 대한 확률

lr = LogisticRegression(C=20, max_iter=1000) # C는 규제 값 기본적으로 1 클수록 규제가 완화, max_iter 반복 횟수
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
print(lr.predict(test_scaled[:5]))
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))
proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))
```

## 점진적인 학습
### 하강법
#### 1. 확률적 경사 하강법
* 훈련세트에서 무작위로 1개를 선택하여 학습 
#### 2. 미니배치 경사 하강법
* 훈련세트에서 여러개씩 선택하여 학습
#### 3. 배치 경사 하강법
* 훈련세트 전체를 선택하여 학습
### 에포크
* 훈련세트를 한번 모두 사용하는 과정
### 손실함수
* 어떤 문제에서 머신러닝 알고리즘이 얼마나 잘못됐는지 측정하는 기준
* 상황에 따라 어떤 손실 함수를 사용할지 선택해야 함
* 샘플 하나에 대한 손실 정의
### 비용함수
* 훈련 세트에 있는 모든 샘플에 대한 손실 함수의 합
```python
import pandas as pd

fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss='log', max_iter=10, random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

sc.partial_fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

import numpy as np

sc = SGDClassifier(loss='log', random_state=42)

train_score = []
test_score = []

classes = np.unique(train_target)

for _ in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes=classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

import matplotlib.pyplot as plt

plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
```

