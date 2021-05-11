# 회귀 알고리즘과 모델 규제

## K-최근접 이웃회귀
* 주어진 데이터에서 가장 가까운 값을 회귀로 에측하는 알고리즘
* 회귀에서는 정확한 값을 맞추는 것은 불가능, 따라서 평가 점수는 결정계수로 표현.
    - 결정계수(R^2) : 타깃과 예측한 값의 차이를 제곱 후, 타깃과 타깃 평균의 차이를 제곱한 값으로 나눈 것.
```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error


perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])


plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight)
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

# K-최근접 이웃 회귀 적용
knr = KNeighborsRegressor()
# 이웃의 개수를 3개로 감소, 이웃의 개수가 줄어들수록 훈련 세트에 더 민감해지고 이웃의 개수를 늘리면 데이터 전반에 있는 일반적 패턴을 따를 것임
# 기본 값은 5
knr.n_neighbors = 3
knr.fit(train_input, train_target)
# 회귀에서 평가는 결정계수라는 점수를 이용하여 평가
print(knr.score(test_input, test_target))
# 테스트 세트에 대한 예측값과 실제 테스트 세트에 타겟을 비교하여 오차 계산
test_prediction = knr.predict(test_input)
mae = mean_absolute_error(test_target, test_prediction)
print(mae)
```

## 선형회귀, 다항회귀 그리고 다중 회귀
### 1. 선형회귀
* 특성과 타깃 사이의 관계를 가장 잘 나타내는 선형 방정식을 찾는다. 
* 특성이 하나면 직선의 방정식도 하나.
* 여러 직선 중 특성을 가장 잘 나타낼 수 있는 직선을 찾아야 함.

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_input, train_target)
print(lr.predict([[50]]))
# lr객체의 가중치(기울기)와 y절편을 구할 수 있음
print(lr.coef_, lr.intercept_)
plt.scatter(train_input, train_target)
plt.plot([15, 50], [15*lr.coef_ + lr.intercept_, 50*lr.coef_ + lr.intercept_])
plt.scatter(50, 1215.7, marker='^')
plt.show()
```

### 2. 다항회귀
* 해당 특성을 이용하여 최적의 곡선을 만들어 내야함.
* 다항식을 사용하여 특성과 타깃 사이의 관계를 나타냄. 
* 함수는 비선형일 수 있지만 선형 회귀로 표현 가능.
```python
from sklearn.linear_model import LinearRegression
# input의 제곱과 input 을 하나의 쌍으로 묶어 열을 두개로 만들기
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.predict([[50**2, 50]]))
# 가중치와 y절편 확인하기
print(lr.coef_, lr.intercept_)

point = np.arange(15, 50)
plt.scatter(train_input, train_target)
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)
plt.scatter([50], [1574], marker='^')
plt.show()
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))
```

### 3. 다중회귀
* 여러개의 특성을 사용하는 회귀 모델.
* 특성이 많으면 선형 모델은 강력한 성능을 발휘.
* 기존의 특성을 이용하여 (특성3 = 특성1 x 특성2)과 같이 새로운 특성을 만들어내는 것을 특성 공학이라고 함.
* `Skitlearn`에서 제공하는 변환기를 이용하여 데이터를 변환할 수 있음.
```python
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge


df = pd.read_csv('https://bit.ly/perch_csv')
perch_full = df.to_numpy()
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)
# PolynomialFeatures 변환기를 이용하여 기존의 특성을 통해 새로운 특성을 만들어냄
poly = PolynomialFeatures(include_bias=False)
# degree는 만들어내는 특성식의 가지수
poly_degree_5 = PolynomialFeatures(degree=5, include_bias=False)
# fit()은 새롭게 만들 특성 조합을 찾고 transform()은 실제 데이터로 변환
# fit_transform()은 위 과정을 한번에 실행
train_poly = poly.fit_transform(train_input)
test_poly = poly.fit_transform(test_input)
# get_feature_name()을 이용하면 각각의 만들어진 특성을 만들어 낸 식 출력
print(poly.get_feature_names())
# poly 객체의 degree 값이 2일 때
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))
# poly 객체의 degree 값이 5일 때
lr.fit(train_poly_5, train_target)
print(lr.score(train_poly_5, train_target))
print(lr.score(test_poly_5, test_target))
```

## 규제
- 규제는 머신러닝 모델이 훈련 세트를 너무 과도하게 학습하지 못하도록 훼방하는 것을 말함.
- 여러개의 특성을 훈련할 때 조금더 보편적인 패턴으로 학습을 시킬 수 있음.
```python
from sklearn.preprocessing import StandardScaler

# 표준점수로 변환해주는 변환기
ss = StandardScaler()
# 변환 시 무조건 PolynomialFeatures로 만든 train_poly를 사용하여 객체를 훈련해야 함
ss.fit(train_poly)
# StandardScaler를 이용하여 표준점수로 변환
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)
```
### 1. 릿지
* 규제가 있는 선형 회귀모델 중 하나.
* 계수를 제곱한 값을 기준으로 규제를 적용.
* 일반적으로 릿지를 선호
* 모델 객체를 만들 때 `alpha` 매개변수를 이용하여 규제의 강도를 조절할 수 있음.
  > * `alpha` 값이 크면 규제 강도가 세지므로 계수 값을 더 줄이고 조금 더 과소적합되도록 유도.   
  > * `alpha` 값이 작으면 계수를 줄이는 역할이 줄어들고 선형 회귀 모델과 점점 유사해지므로 과대적합될 가능성이 큼.
```python
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

train_score = []
test_score = []

# 알파 계수 별 성능 확인하기
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # 릿지 모델 만들기
    ridge = Ridge(alpha=alpha)
    # 릿지 모델 훈련
    ridge.fit(train_scaled, train_target)
    # 훈련 점수와 테스트 점수 리스트에 저장
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))

print(train_score)
print(test_score)
```
### 2. 라쏘
* 계수의 절대값을 기준으로 규제를 적용.
```python
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
```

