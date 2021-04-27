# 비지도 학습

## 군집 알고리즘
* 비슷한 샘플끼리 하나의 그룹으로 모으는 대표적인 비지도 학습 작업. 이를 통해 얻은 샘플 그룹을 클러스터라고 부름.
```python
import numpy as np
import matplotlib.pyplot as plt

fruits = np.load('fruits_300.npy', allow_pickle=True)
fig, axs = plt.subplots(1, 3)

apple = fruits[0:100].reshape(-1, 100*100)
banana = fruits[100:200].reshape(-1, 100*100)
pineapple = fruits[200:300].reshape(-1, 100*100)

apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)

axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(banana_mean, cmap='gray_r')
axs[2].imshow(pineapple_mean, cmap='gray_r')
plt.show()

abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis=(1,2))

apple_index = np.argsort(abs_mean)[:100]
print(apple_index)
fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
  for j in range(10):
    axs[i, j].imshow(fruits[apple_index[i*10 + j]], cmap='gray_r')
    axs[i, j].axis('off')
plt.show()
```
## K-Mean
* 처음에 랜덤하게 클러스터 중심을 정하고 클러스터를 만듬. 그다음 클러스터의 중심을 이동하고 다시 클러스터를 만드는 식으로 반복하여 최적의 클러스터를 구성
* 클러스터 중심은 K-Mean 알고리즘이 만든 클러스터에 속한 샘플의 특성 평균값을 뜻함.
### 엘보우 방법
* 최적의 클러스터 개수를 정하는 방법 중 하나.
* 이너셔는 클러스터 중심과 샘플 사이 거리의 제곱 합.
* 클러스터 개수에 따라 이너셔 감소가 꺽이는 지점이 적절한 클러스터 개수 K임.
```python
import numpy as np
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)
print(km.labels_)
print(np.unique(km.labels_, return_counts=True))

import matplotlib.pyplot as plt
def draw_fruits(arr, ratio=1):
  n = len(arr)
  rows = int(np.ceil(n/10))
  cols = n if rows < 2 else 10
  fig, axs = plt.subplots(rows, cols, figsize=(cols*ratio, rows*ratio), squeeze=False)
  for i in range(rows):
    for j in range(cols):
      if i*10 + j < n:
        axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
      axs[i, j].axis('off')
  plt.show()

draw_fruits(fruits[km.labels_==2])
print(km.predict(fruits_2d[100:101]))
inertia = []
for k in range(2, 7):
  km = KMeans(n_clusters=k, random_state=42)
  km.fit(fruits_2d)
  inertia.append(km.inertia_)
plt.plot(range(2, 7), inertia)
plt.show()
```
## 주성분 분석
### 차원 축소
* 원본 데이터의 특성을 적은 수의 새로운 특성으로 변환하는 비지도 학습의 한 종류.
* 차원 축소는 저장 공간을 줄이고 시각화하기 쉬움. 또한 다른 알고리즘의 성능을 향상 시킴.
### 주성분 분석
* 차원 축소 알고리즘의 하나로 데이터에서 가장 분산이 큰 방향을 찾는 방법. 
* 이런 방향을 주성분이라고 부르고 원본 데이터를 주성분에 투영하여 새로운 특성을 만들 수 있음.
```python
import numpy as np
fruits = np.load('fruits_300.npy')

from sklearn.decomposition import PCA
pca = PCA(n_components=50)
fruits_2d = fruits.reshape(-1, 100*100)
pca.fit(fruits_2d)
fruits_pca = pca.transform(fruits_2d)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

target = np.array([0]*100 + [1]*100 + [2]*100)
from sklearn.model_selection import cross_validate
scores = cross_validate(lr, fruits_2d, target)
print(np.mean(scores['test_score']))

scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))

pca = PCA(n_components=0.5)
pca.fit(fruits_2d)
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
```