#!/usr/bin/env python
# coding: utf-8

# ### 기본 미분 필터

# In[4]:


# 미분 커널로 경계 검출 (edge_differential.py)

import cv2
import numpy as np

img = cv2.imread("C:/cdd/223.jpg")

# 미분 커널 생성
gx_kernel = np.array([[-1,1]])
gy_kernel = np.array([[-1],[1]])

# 필터 적용
edge_gx = cv2.filter2D(img, -1, gx_kernel)
edge_gy = cv2.filter2D(img, -1, gy_kernel)

#결과 출력
merged = np.hstack((img, edge_gx, edge_gy))
cv2.imshow('edge', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 로버츠 교차 필터

# In[5]:


# 로버츠 교차 필터를 적용한 경계 검출 (edge_roberts.py)

import cv2
import numpy as np

img = cv2.imread("C:/cdd/223.jpg")

# 로버츠 커널 생성 ---①
gx_kernel = np.array([[1,0], [0,-1]])
gy_kernel = np.array([[0, 1],[-1,0]])

# 커널 적용 ---② 
edge_gx = cv2.filter2D(img, -1, gx_kernel)
edge_gy = cv2.filter2D(img, -1, gy_kernel)

# 결과 출력
merged = np.hstack((img, edge_gx, edge_gy, edge_gx+edge_gy))
cv2.imshow('roberts cross', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 프리윗 필터

# In[ ]:


# 프리윗 마스크를 적용한 경계 검출 (edge_prewitt.py)

import cv2
import numpy as np

file_name = "../img/sudoku.jpg"
img = cv2.imread(file_name)

# 프리윗 커널 생성
gx_k = np.array([[-1,0,1], [-1,0,1],[-1,0,1]])
gy_k = np.array([[-1,-1,-1],[0,0,0], [1,1,1]])

# 프리윗 커널 필터 적용
edge_gx = cv2.filter2D(img, -1, gx_k)
edge_gy = cv2.filter2D(img, -1, gy_k)

# 결과 출력
merged = np.hstack((img, edge_gx, edge_gy, edge_gx+edge_gy))
cv2.imshow('prewitt', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 소벨 필터 (Sobel Filter)

# In[ ]:


'''

dst = cv2.Sobel(src, ddepth, dx, dy, dst, ksize, scale, delta, borderType)
src: 입력 영상
ddepth: 출력 영상의 dtype (-1: 입력 영상과 동일)
dx, dy: 미분 차수 (0, 1, 2 중 선택, 둘 다 0일 수는 없음)
ksize: 커널의 크기 (1, 3, 5, 7 중 선택)
scale: 미분에 사용할 계수
delta: 연산 결과에 가산할 값

'''


# In[6]:


# 소벨 마스크를 적용한 경계 검출 (edge_sobel.py)

import cv2
import numpy as np

img = cv2.imread("C:/cdd/223.jpg")

# 소벨 커널을 직접 생성해서 엣지 검출 ---①
## 소벨 커널 생성
gx_k = np.array([[-1,0,1], [-2,0,2],[-1,0,1]])
gy_k = np.array([[-1,-2,-1],[0,0,0], [1,2,1]])
## 소벨 필터 적용
edge_gx = cv2.filter2D(img, -1, gx_k)
edge_gy = cv2.filter2D(img, -1, gy_k)

# 소벨 API를 생성해서 엣지 검출
sobelx = cv2.Sobel(img, -1, 1, 0, ksize=3)
sobely = cv2.Sobel(img, -1, 0, 1, ksize=3) 

# 결과 출력
merged1 = np.hstack((img, edge_gx, edge_gy, edge_gx+edge_gy))
merged2 = np.hstack((img, sobelx, sobely, sobelx+sobely))
merged = np.vstack((merged1, merged2))
cv2.imshow('sobel', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 샤르필터

# In[ ]:


# 샤르 마스크를 적용한 경계 검출 (edge_scharr.py)

import cv2
import numpy as np

img = cv2.imread("../img/sudoku.jpg")

# 샤르 커널을 직접 생성해서 엣지 검출 ---①
gx_k = np.array([[-3,0,3], [-10,0,10],[-3,0,3]])
gy_k = np.array([[-3,-10,-3],[0,0,0], [3,10,3]])
edge_gx = cv2.filter2D(img, -1, gx_k)
edge_gy = cv2.filter2D(img, -1, gy_k)

# 샤르 API로 엣지 검출 ---②
scharrx = cv2.Scharr(img, -1, 1, 0)
scharry = cv2.Scharr(img, -1, 0, 1)

# 결과 출력
merged1 = np.hstack((img, edge_gx, edge_gy))
merged2 = np.hstack((img, scharrx, scharry))
merged = np.vstack((merged1, merged2))
cv2.imshow('Scharr', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 라플라시안 필터 (Laplacian Filter)

# In[7]:


# 라플라시안 마스크를 적용한 경계 검출 (edge_laplacian.py)

import cv2
import numpy as np

img = cv2.imread("../img/sudoku.jpg")

# 라플라시안 필터 적용 ---①
edge = cv2.Laplacian(img, -1)

# 결과 출력
merged = np.hstack((img, edge))
cv2.imshow('Laplacian', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 캐니 엣지

# In[ ]:


'''

캐니 엣지는 지금까지 살펴본 것처럼 한 가지 필터만 사용하는 것이 아니라 다음의 4단계 알고리즘에 따라 경계를 검출합니다. 

1. 노이즈 제거: 5 x 5 가우시안 블러링 필터로 노이즈 제거
2. 경계 그레디언트 방향 계산: 소벨 필터로 경계 및 그레디언트 방향 검출
3. 비최대치 억제(Non-Maximum Suppression): 그레디언트 방향에서 검출된 경계 중 가장 큰 값만 선택하고 나머지는 제거
4. 이력 스레시홀딩: 두 개의 경계 값(Max, Min)을 지정해서 경계 영역에 있는 픽셀들 중 큰 경계 값(Max) 밖의 픽셀과 연결성이 없는 픽셀 제거

OpenCV에서 제공하는 캐니 엣지는 함수는 아래와 같습니다.

edges = cv2.Canny(img, threshold1, threshold2, edges, apertureSize, L2gardient)
img: 입력 영상
threshold1, threshold2: 이력 스레시홀딩에 사용할 Min, Max 값
apertureSize: 소벨 마스크에 사용할 커널 크기
L2gradient: 그레디언트 강도를 구할 방식 (True: 제곱 합의 루트 False: 절댓값의 합)
edges: 엣지 결과 값을 갖는 2차원 배열

'''


# In[12]:


# 캐니 엣지 검출 (edge_canny.py)

import cv2, time
import numpy as np

img = cv2.imread("C:/cdd/223.jpg")

# 케니 엣지 적용 
edges = cv2.Canny(img,100,255)

# 결과 출력
cv2.imshow('Original', img)
cv2.imshow('Canny', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




