#!/usr/bin/env python
# coding: utf-8

# ### 침식연산

# In[ ]:


'''

침식 연산은 이밎를 갂아 내는 연산을 뜻한다.
구조화 요소 커널이라는 0과 1로 구성된 커널이 필요함.


조화 요소 커널 생성을 위한 함수는 다음과 같습니다.

cv2.getStructuringElement(shape, ksize, anchor)
shape: 구조화 요소 커널 모양 (cv2.MORPH_RECT: 사각형, cv2.MORPH_EPLIPSE: 타원형, cv2.MORPH_CROSS: 십자형)
ksize: 커널 크기
anchor(optional): 구조화 요소의 기준점, cv2.MORPH_CROSS에만 의미 있으며 기본 값은 중심점 (-1, -1)
위 함수로 생성한 구조화 요소 커널로 침식 연산을 수행하는 함수는 다음과 같습니다.

dst = cv2.erode(src, kernel, anchor, iterations, borderType, borderValue)
src: 입력 영상, 바이너리
kernel: 구조화 요소 커널
anchor(optional): cv2.getStructuringElement()와 동일
iterations(optional): 침식 연산 적용 반복 횟수
boderType(optional): 외곽 영역 보정 방법 
boderValue(optional): 외곽 영역 보정 값

'''


# In[ ]:


# 침식 연산 (morph_erode.py)

import cv2
import numpy as np

img = cv2.imread('../img/morph_dot.png')

# 구조화 요소 커널, 사각형 (3x3) 생성 ---①
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# 침식 연산 적용 ---②
erosion = cv2.erode(img, k)

# 결과 출력
merged = np.hstack((img, erosion))
cv2.imshow('Erode', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 팽창 연산

# In[ ]:


# 팽창 연산 (morph_dilate.py)

import cv2
import numpy as np

img = cv2.imread('../img/morph_hole.png')

# 구조화 요소 커널, 사각형 (3x3) 생성 ---①
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# 팽창 연산 적용 ---②
dst = cv2.dilate(img, k)

# 결과 출력
merged = np.hstack((img, dst))
cv2.imshow('Dilation', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 열림, 닫힘, 그레디언트, 탑햇, 블랙햇 연산

# In[ ]:


'''

침식과 팽창의 연산을 조합하면 원래의 모양을 유지하면서 노이즈를 제거하는 효과를 거둘 수 있습니다.


침식 연산 후 팽창 연산을 적용하는 것을 열림(openning) 연산 :

-> 주변보다 밝은 노이즈 제거 효과적, 독립된 개체 분리, 돌출된 모양을 제거하는데 효과적


팽창 연산 후 침식 연산을 적용하는 것을 닫힘(closing) 연산 :

-> 주변보다 어두운 조니즈를 제거하는데 효과적이면서 끊어져 보니는 개체를 연결하거나 구멍을 메우는데 효과적



팽창 연산을 적용한 이미지에서 침식 연산을 적용한 이미지를 빼면 경계 픽셀만 얻게 되는데, 
이는 앞서 살펴본 경계 검출과 비슷합니다. 이런 연산을 그레디언트(gradient) 연산이라고 합니다.

그레디언트 = 팽창 - 침식

또한, 원본에서 열림 연산 적용 결과를 빼면 값이 크게 튀는 밝은 영역을 강조할 수 있고, 닫힘 연산 적용 결과에서 원본을 빼면 어두운 부분을 강조할 수 있습니다. 이것을 각각 탑햇(top hat)과 블랙햇(black hat) 연산이라고 합니다.

탑햇 = 원본 - 열림
블랙햇 = 닫힘 - 원본

'''


# In[ ]:


'''

penCV는 열림, 닫힘, 그레디언트, 탑햇, 블랙햇 연산을 위해서 아래의 함수를 제공합니다.

dst = cv2.morphologyEx(src, op, kernel, dst, anchor, iteration, borderType, borderValue)
src: 입력 영상
op: 모폴로지 연산 종류 (cv2.MORPH_OPEN: 열림 연산, cv2.MORPH_COLSE: 닫힘 연산, cv2.MORPH_GRADIENT: 그레디언트 연산, cv2.MORPH_TOPHAT: 탑햇 연산, cv2.MORPH_BLACKHAT: 블랙햇 연산)
kernel: 구조화 요소 커널
dst(optional): 결과 영상
anchor(optional): 커널의 기준점
iteration(optional): 연산 반복 횟수
borderType(optional): 외곽 영역 보정 방법
borderValue(optional): 외곽 영역 보정 값

'''


# In[1]:


# 열림 현상과 닫힘 연산으로 노이즈 제거

import cv2
import numpy as np

img1 = cv2.imread('C:/cdd/232.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('C:/cdd/skull.jpg', cv2.IMREAD_GRAYSCALE)    

# 구조화 요소 커널, 사각형 (5x5) 생성 ---①
k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# 열림 연산 적용 ---②
opening = cv2.morphologyEx(img1, cv2.MORPH_OPEN, k)
# 닫힘 연산 적용 ---③
closing = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, k)

# 결과 출력
merged1 = np.hstack((img1, opening))
merged2 = np.hstack((img2, closing))
merged3 = np.vstack((merged1, merged2))
cv2.imshow('opening, closing', merged3)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[2]:


# 모폴로지 탑햇, 블랙햇 연산 (morph_hat.py)

import cv2
import numpy as np

img = cv2.imread('C:/cdd/Full_Moon.jpg')

# 구조화 요소 커널, 사각형 (5x5) 생성 ---①
k = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
# 탑햇 연산 적용 ---②
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k)
# 블랫햇 연산 적용 ---③
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, k)

# 결과 출력
merged = np.hstack((img, tophat, blackhat))
cv2.imshow('tophat blackhat', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




