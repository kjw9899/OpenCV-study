#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''

공간 영역 필터링(special domain filtering) : 새로운 픽셀 값을 얻을 때 하나의 픽셀값이 아닌 그 주변의 픽셀 값을 활용하는 방법
블러핑(Blerring) : 기존의 영상을 흐릿하게 만드는 작업

'''


# ### 필터(filter)와 컨볼루션(Convolution)

# In[ ]:


'''

공간 영역 필터링은 연산 대상 픽셀과 그 주변 픽셀들을 활용하여 새로운 픽셀 값을 얻는 방법이라고 했습니다.
이때 주변 픽셀을 어느 범위까지 활용할지 그리고 연산은 어떻게 할지를 결정해야 합니다.
이런 역할을 하는 것이 바로 커널(kernel)입니다.
커널은 윈도(window), 필터(filter), 마스크(mask)라고도 부릅니다. 아래 그림에서 가운데 있는 3 x 3 짜리 행렬이 바로 커널입니다. 

'''


# In[ ]:


'''

OpenCV에서는 아래 함수로 컨볼루션 연산을 지원합니다.

dst = cv2.filter2D(src, ddepth, kernel, dst, anchor, delta, borderType)
src: 입력 영상, Numpy 배열
ddepth: 출력 영상의 dtype (-1: 입력 영상과 동일)
kernel: 컨볼루션 커널, float32의 n x n 크기 배열
dst(optional): 결과 영상
anchor(optional): 커널의 기준점, default: 중심점 (-1, -1)
delta(optional): 필터가 적용된 결과에 추가할 값
borderType(optional): 외곽 픽셀 보정 방법 지정

'''


# ### 평균블러링 (Average Blurring)

# In[3]:


# 평균 필터를 생성하여 블러 적용

import cv2
import numpy as np

img = cv2.imread('C:/cdd/232.jpg')
'''

[5,5] 평균 필터 생설
kernel = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                   [0.04, 0.04, 0.04, 0.04, 0.04],
                   [0.04, 0.04, 0.04, 0.04, 0.04],
                   [0.04, 0.04, 0.04, 0.04, 0.04],
                   [0.04, 0.04, 0.04, 0.04, 0.04]])

'''
# [5,5]  평균 필터 커널 생성
kernel = np.ones((5,5))/5**2
# 필터 적용
blured = cv2.filter2D(img, -1, kernel)

# 결과 출력
cv2.imshow('origin', img)
cv2.imshow('avrg blur', blured)
cv2.waitKey()
cv2.destroyAllWindows()


# In[ ]:


'''

 개발자가 직접 커널을 생성하지 않고도 평균 블러링을 적용할 수 있습니다. OpenCV에서는 아래와 같은 평균 블러링 함수를 제공합니다.

dst = cv2.blur(src, ksize, dst, anchor, borderType)
src: 입력 영상, numpy 배열
ksize: 커널의 크기
나머지 파라미터는 cv2.filter2D()와 동일

dst = cv2.boxFilter(src, ddepth, ksize, dst, anchor, normalize, borderType)
ddepth: 출력 영상의 dtype (-1: 입력 영상과 동일)
normalize(optional): 커널 크기로 정규화(1/ksize²) 지정 여부 (Boolean), default=True
나머지 파라미터는 cv2.filter2D()와 동일

'''


# In[4]:


# 블러 전용 함수로 블러링 적용

import cv2
import numpy as np

file_name = 'C:/cdd/232.jpg'
img = cv2.imread(file_name)

# blur() 함수로 블러링  ---①
blur1 = cv2.blur(img, (10,10))
# boxFilter() 함수로 블러링 적용 ---②
blur2 = cv2.boxFilter(img, -1, (10,10))

# 결과 출력
merged = np.hstack( (img, blur1, blur2))
cv2.imshow('blur', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 가우시안 블러링

# In[ ]:


'''

가우시안 블러링 커널을 적용하면 대상 픽셀에 가까울수록 많은 영향을 주고, 
멀어질수록 적은 영향을 주기 때문에 원래의 영상과 비슷하면서도 노이즈를 제거하는 효과가 있습니다.

'''


# In[ ]:


'''

OpenCV에서는 아래와 같이 가우시안 블러링을 적용하는 함수를 제공합니다.

cv2.GaussianBlur(src, ksize, sigmaX, sigmaY, borderType)
src: 입력 영상
ksize: 커널 크기 (주로 홀수)
sigmaX: X 방향 표준편차 (0: auto)
sigmaY(optional): Y 방향 표준편차 (default: sigmaX)
borderType(optional): 외곽 테두리 보정 방식

ret = cv2.getGaussianKernel(ksize, sigma, ktype)
ret: 가우시안 커널 (1차원이므로 ret * ret.T 형태로 사용해야 함)

cv2.GaussianBlur() 함수는 커널 크기와 표준 편차를 전달하면 가우시안 블러링을 적용해줍니다.
sigmaX에 0을 전달하면 자동으로 표준편차를 선택해서 사용하고, sigmaY를 생략하면 sigmaX 값과 동일하게 적용합니다.

cv2.getGaussianKernel() 함수는 커널 크기와 표준 편차를 전달하면 가우시안 필터를 반환합니다.
반환된 필터는 1차원이므로 cv2.filter2D() 함수에 사용하려면 ret * ret.T와 같은 형식으로 전달해야 합니다.

'''


# In[2]:


# 가우시안 블러링

import cv2
import numpy as np

img = cv2.imread('C:/cdd/232.jpg')

 #가우시안 커널을 직접 생성해서 블러링  ---①
k1 = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]]) *(1/16)
blur1 = cv2.filter2D(img, -1, k1)

# 가우시안 커널을 API로 얻어서 블러링 ---②
k2 = cv2.getGaussianKernel(3, 0)
blur2 = cv2.filter2D(img, -1, k2*k2.T)

# 가우시안 블러 API로 블러링 ---③
blur3 = cv2.GaussianBlur(img, (3, 3), 0)

# 결과 출력
print('k1:', k1)
print('k2:', k2*k2.T)
merged = np.hstack((img, blur1, blur2, blur3))
cv2.imshow('gaussian blur', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 미디언 블러링

# In[ ]:


'''

dst = cv2.medianBlur(src, ksize)
src: 입력 영상
ksize: 커널 크기

'''


# In[ ]:


# 미디언 블러링 (blur_median.py)

import cv2
import numpy as np

img = cv2.imread("../img/salt_pepper_noise.jpg")

# 미디언 블러 적용 --- ①
blur = cv2.medianBlur(img, 5)

# 결과 출력 
merged = np.hstack((img,blur))
cv2.imshow('media', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


# 미디언 블러링 (blur_median.py)

import cv2
import numpy as np

img = cv2.imread("../img/salt_pepper_noise.jpg")

# 미디언 블러 적용 --- ①
blur = cv2.medianBlur(img, 5)

# 결과 출력 
merged = np.hstack((img,blur))
cv2.imshow('media', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 바이레터럴 필터(Bilateral Filter)

# In[ ]:


'''

dst = cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace, dst, borderType)
src: 입력 영상
d: 필터의 직경(diameter), 5보다 크면 매우 느림
sigmaColor: 색공간의 시그마 값
sigmaSpace: 좌표 공간의 시그마 값

'''


# In[ ]:


# 바이레터럴 필터와 가우시안 필터 비교 (blur_bilateral.py)

import cv2
import numpy as np

img = cv2.imread("../img/gaussian_noise.jpg")

# 가우시안 필터 적용 ---①
blur1 = cv2.GaussianBlur(img, (5,5), 0)

# 바이레터럴 필터 적용 ---②
blur2 = cv2.bilateralFilter(img, 5, 75, 75)

# 결과 출력
merged = np.hstack((img, blur1, blur2))
cv2.imshow('bilateral', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()

#출처 : https://bkshin.tistory.com/entry/OpenCV-17-%ED%95%84%ED%84%B0Filter%EC%99%80-%EC%BB%A8%EB%B3%BC%EB%A3%A8%EC%85%98Convolution-%EC%97%B0%EC%82%B0-%ED%8F%89%EA%B7%A0-%EB%B8%94%EB%9F%AC%EB%A7%81-%EA%B0%80%EC%9A%B0%EC%8B%9C%EC%95%88-%EB%B8%94%EB%9F%AC%EB%A7%81-%EB%AF%B8%EB%94%94%EC%96%B8-%EB%B8%94%EB%9F%AC%EB%A7%81-%EB%B0%94%EC%9D%B4%EB%A0%88%ED%84%B0%EB%9F%B4-%ED%95%84%ED%84%B0?category=1148027
