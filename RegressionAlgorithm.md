# 회귀 알고리즘과 모델 규제

## K-최근접 이웃회귀
* 주어진 데이터에서 가장 가까운 값을 회귀로 에측하는 알고리즘
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
knr.n_neighbors = 3
knr.fit(train_input, train_target)
print(knr.score(test_input, test_target))
print(knr.score(train_input, train_target))
print(knr.predict([[50]]))
```

## 선형회귀, 다항회귀 그리고 다중 회귀
### 1. 선형회귀
* 특성과 타깃 사이의 관계를 가장 잘 나타내는 선형 방정식을 찾는다. 
* 특성이 하나면 직선의 방정식도 하나.
```
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_input, train_target)
print(lr.predict([[50]]))
plt.scatter(train_input, train_target)
plt.plot([15, 50], [15*lr.coef_ + lr.intercept_, 50*lr.coef_ + lr.intercept_])
plt.scatter(50, 1215.7, marker='^')
plt.show()
```

### 2. 다항회귀
* 다항식을 사용하여 특성과 타깃 사이의 관계를 나타냄. 
* 함수는 비선형일 수 있지만 선형 회귀로 표현 가능.
```python
from sklearn.linear_model import LinearRegression
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.predict([[50**2, 50]]))

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
```python
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge


df = pd.read_csv('https://bit.ly/perch_csv')
perch_full = df.to_numpy()
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)
poly = PolynomialFeatures(degree=5, include_bias=False)
train_poly = poly.fit_transform(train_input)
test_poly = poly.fit_transform(test_input)

ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

train_score = []
test_score = []

alpah_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpah_list:
  ridge = Ridge(alpha=alpha)
  ridge.fit(train_scaled, train_target)
  train_score.append(ridge.score(train_scaled, train_target))
  test_score.append(ridge.score(test_scaled, test_target))

print(train_score)
print(test_score)
```

## 릿지와 라쏘, 하이퍼파라미터
### 1. 릿지
* 규제가 있는 선형 회귀모델 중 하나.
* 선형모델의 계수를 작게 만들어 과대적합을 완화시킴.
### 2. 라쏘
* 릿지와 달리 계수 값을 0으로 만들 수 있음.
### 3. 하이퍼파라미터
* 머신러닝 알고리즘이 학습하지 않은 파라미터, 이런 파라미터는 사람이 사전에 지정해야함.

