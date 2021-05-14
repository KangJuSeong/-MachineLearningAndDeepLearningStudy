# 트리 알고리즘

## 결정트리
* 예/아니요에 대한 질문을 이어나가면서 정답을 찾아 학습하는 알고리즘. 예측과정을 이해하기 쉽고 성능도 뛰어남.
* 가지치기를 통해 자라날 수 있는 트리의 최대 깊이를 지정할 수 있음.
* 결정 트리에서는 데이터의 표준화가 필요 없음.
1. 불순도
    * 결정트리가 최적의 질문을 찾기 위한 기준.
    * `DecisionTreeClassifier`클래스의 `criterion`매개변수의 기본값은 `gini`
    * gini 불순도는 클래스의 비율을 제곱해서 더한 다음 1에서 빼면 됨.
      > GINI 불순도 =  1 - (음성 클래스 비율의 제곱 + 양성 클래스 비율의 제곱)
    * `criterion = 'entropy'`를 지정하여 엔트로피 불순도를 사용할 수 있음.
      > ENTROPY 불순도 = -음성 클래스 비율 x log2(음성 클래스 비율) - 양성 클래스 비율 x log2(양성 클래스 비율)

2. 정보 이득
    * 부모 노드와 자식 노드의 불순도 차이.
    * 결정트리는 정보이득이 최대화되도록 학습함.

3. 특성 중요도
    * 결정트리에 사용된 특성이 불순도를 감소하는데 기여한 정도를 나타내는 값. 특성 중요도를 검사할 수 있는것이 결정트리의 장점.
```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


wine = pd.read_csv('wine.csv')
input_data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target_data = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(input_data, target_data, random_state=42, test_size=0.2)
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 결정트리 객체 생성
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

# 결정트리 그래프로 출력
plt.figure(figsize=(10, 7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

# 가지치기 -> 트리의 깊이를 정하기
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_input, train_target))
print(dt.socre(test_input, test_target))

# 해당 트리에서 어떤 특성이 유용한지 각 특성별 중요도 출력
print(dt.feature_importances_)
```
## 교차 검증과 그리드 검증
* 테스트 세트를 사용하지 않으면 모델이 과대적합인지 과소적합인지 판단하기 어려움.
* 테스트 세트를 사용하지 않고 이를 측정하는 간단한 방법은 훈련 세트 내에서 나눠서 검증 세트를 만드는 것.
### 1. 교차 검증
* 검증 세트를 만들려면 훈련 세트가 줄어들고, 검증 세트를 줄이면 검증 점수가 불안정하므로 안정적인 검증 점수와 훈련에 더 많은 데이터를 사용할 수 있는 교차 검증을 사용.
* 훈련 세트를 여러 폴드로 나눈 다음 한 폴드가 검증 세트의 역할을 하고 나머지 폴드에서 모델을 훈련.
```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import numpy as np


wine = pd.read_csv('wine.csv')
input_data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target_data = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(input_data, target_data, random_state=42, test_size=0.2)

dt = DecisionTreeClassifier(random_state=42)
# cv를 통해 폴드 수를 지정할 수 있음
scores = cross_validate(dt, train_input, train_target, cv=5)
# fit_time, score_time 은 모델을 훈련하는 시간과 검증하는 시간, test_score 는 각각의 폴드를 통해 나온 검증 점수
print(scores)
# 검증 폴드의 점수의 평균으로 검증 점수를 출력
print(np.mean(scores['test_score']))

# 교차 검증 시 훈련세트를 섞으려면 분할기를 지정해야 함
# 회귀 모델 = KFold, 분류 모델 = StratifiedKFold 사용
splitter = StratifiedKFold(n_splits=10,  # 폴드 수 지정
                           shuffle=True, # 셔플 여부 
                           random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))
```
### 2. 그리드 검증
* 하이퍼파라미터 탐색을 자동화해 주는 도구.
  > 하이퍼파라미터
  > * 모델이 학습할 수 없어서 사용자가 지정해야만 하는 파라미터
* 탐색할 매개변수를 나열하면 교차 검증을 수행하여 가장 좋은 검증점수의 매개변수 조합을 찾음.
* 매개변수의 값이 수치일 때 값의 범위나 간격을 미리 정하기 어려울 때, 너무 많은 매개변수 조건이 있어 수행시간이 오래걸릴 때 랜덤 서치의 효율이 좋음.
```python
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np

wine = pd.read_csv('wine.csv')

input_data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target_data = wine['class'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(input_data, target_data, random_state=42, test_size=0.2)
# 매개변수에 들어갈 값들을 dict 형태로 설정
params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001), 'max_depth': range(5, 20, 1), 'min_samples_split': range(2, 100, 10)}
# n_jobs 는 연산간 사용할 cpu 코어의 수 (-1은 모든 cpu 사용), 검증시 cv 의 기본값은 5(교차 검증 폴드 수)
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
# 검증 점수가 가장 높은 모델 선택
dt = gs.best_estimator_
# 검증 점수가 높은 모델을 학습할 때 사용한 매개변수의 값들을 출력
print(gs.best_params_)
# 각각의 모델을 교차검증하여 나온 검증 점수의 평균 출력
print(gs.cv_results_['mean_test_score'])
# 가장 높은 검증 점수의 인덱스를 추출 후 해당 모델의 매개변수 값 출력
best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index])

print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

# 랜덤 서치
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV

params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth': randint(20, 50),
          'min_samples_split': randint(2, 25),
          'min_samples_leaf': randint(1, 25)}
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input, train_target)
print(gs.best_params_)
dt = gs.best_estimator_
print(dt.score(test_input, test_target))
```
## 트리의 앙상블
### 앙상블 학습
* 정형 데이터를 다루는 데 가장 뛰어난 성과를 내는 알고리즘
* 더 좋은 예측 결과를 만들기 위해 여러 개의 모델을 훈련함. 
#### 1. 랜덤 포레스트
* 대표적인 앙상블 학습
* 결정 트리를 랜덤하게 만들어 결정 트리의 숲을 만드는 개념 또한 각 결정 트리의 예측을 사용회 최종 예측을 만듬.  
* 부트스트랩 샘플을 사용하고 랜덤하게 일부 특성을 선택하여 트리를 만듬.
>- 부트스트랩 샘플
>   * 입력한 훈련 데이터에서 중복을 허용하여 랜덤하게 샘플을 추출하여 훈련데이터를 만드는 샘플링 기법.
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
