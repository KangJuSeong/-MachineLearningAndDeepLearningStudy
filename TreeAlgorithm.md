# 트리 알고리즘

## 결정트리
* 예/아니요에 대한 질문을 이어나가면서 정답을 찾아 학습하는 알고리즘. 예측과정을 이해하기 쉽고 성능도 뛰어남.
### 불순도
* 결정트리가 최적의 질문을 찾기 위한 기준.
### 정보 이득
* 부모 노드와 자식 노드의 불순도 차이. 결정트리는 정보이득이 최대화되도록 학습함.
### 특성 중요도
* 결정트리에 사용된 특성이 불순도를 감소하는데 기여한 정도를 나타내는 값. 특성 중요도를 검사할 수 있는것이 결정트리의 장점.
```python
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine-date')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l2')
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
```
## 교차 검증과 그리드 검증
### 1. 교차 검증
* 훈련 세트를 여러 폴드로 나눈 다음 한 폴드가 검증 세트의 역할을 하고 나머지 폴드에서 모델을 훈련.
### 2. 그리드 검증
* 하이퍼파라미터 탐색을 자동화해 주는 도구.
* 탐색할 매개변수를 나열하면 교차 검증을 수행하여 가장 좋은 검증점수의 매개변수 조합을 찾음.
```python
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine-date')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
# dt.fit(sub_input, sub_target)
# print(dt.score(sub_input, sub_target))
# print(dt.score(val_input, val_target))

from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
import numpy as np

splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))

from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001), 'max_depth': range(5, 20, 1), 'min_samples_split': range(2, 100, 10)}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
dt = gs.best_estimator_
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))
print(gs.best_params_)

from scipy.stats import uniform, randint
params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth': randint(20, 50),
          'min_samples_split': randint(2, 25),
          'min_samples_leaf': randint(1, 25)}

from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input, train_target)
print(gs.best_params_)
dt = gs.best_estimator_
print(dt.score(test_input, test_target))
```
## 트리의 앙상블
### 앙상블 학습
* 더 좋은 예측 결과를 만들기 위해 여러 개의 모델을 훈련하는 머신러닝 알고리즘
#### 1. 랜덤 포레스트
* 대표적인 결정트리. 
* 부트스트랩 샘플을 사용하고 랜덤하게 일부 특성을 선택하여 트리를 만듬.
>- 부트스트랩 샘플
>   * 입력한 훈련 데이터에서 랜덤하게 샘플을 추출하여 훈련데이터를 만드는 샘플링 기법.
```python
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier # 랜덤 포레스트 (부트스트랩 샘플링)


rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
rf.fit(train_input, train_target)
print(rf.feature_importances_)
```
#### 2. 엑스트라 포레스트
* 랜덤 포레스트와 비슷하게 결정 트리를 사용.
* 부트스트랩 샘플링을 사용하지 않고 랜덤하게 노드를 분할해 과대적합을 감소시킴.
```python
from sklearn.ensemble import ExtraTreesClassifier # 엑스트라 포레스트


et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(et, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
et.fit(train_input, train_target)
print(et.feature_importances_)
```
#### 3. 그레이디언트 부스팅
* 랜덤 포레스트와 엑스트라 포레스트와 달리 결정트리를 연속적으로 추가하여 손실 함수를 최소화하는 앙상블 방법.
* 훈련속도가 느리지만 성능은 좋음
```python
from sklearn.ensemble import GradientBoostingClassifier # 그레이디언트 부스팅


gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
gb.fit(train_input, train_target)
print(gb.feature_importances_)
```
#### 4. 히스토그램 기반 그레디언트 부스팅
* 그레이디언트 부스팅의 속도를 개선하여 안정적인 결과와 높은 성능을 띔.
```python
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier # 히스토그램기반 그레이디언트 부스팅


hgb = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hgb, train_input, train_target, return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```
