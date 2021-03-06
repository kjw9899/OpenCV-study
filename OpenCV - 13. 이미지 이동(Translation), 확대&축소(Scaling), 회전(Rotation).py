#!/usr/bin/env python
# coding: utf-8

# ### 이미지 이동

# In[ ]:


'''

이미지 이동 ? 간단하다. 원해 잇던 좌표에 이동시키려는 거리만큼 더하면 된다.

 이미지의 좌표를 이동하는 변환 행렬은 2 x 3 행렬입니다. 
 변환 행렬이란 어떤 좌표를 선형 변환(linear tranformations) 해주는 행렬을 뜻합니다.
 쉽게 말해서 어떤 좌표를 다른 좌표로 이동시켜주는 행렬이라는 뜻입니다. 
 즉, 어떤 좌표에 변환 행렬을 곱해주면 다른 좌표가 구해지는 행렬입니다. 아래는 이미지 이동에 대한 변환 행렬입니다.

dst = cv2.warpAffine(src, matrix, dsize, dst, flags, borderMode, borderValue)
src: 원본 이미지, numpy 배열
matrix: 2 x 3 변환행렬, dtype=float32
dsize: 결과 이미지의 크기, (width, height)
flags(optional): 보간법 알고리즘 플래그
borderMode(optional): 외곽 영역 보정 플래그
borderValue(optional): cv2.BORDER_CONSTANT 외곽 영역 보정 플래그일 경우 사용할 색상 값 (default=0)
dst: 결과 이미지

flags의 값은 아래와 같습니다.
cv2.INTER_LINEAR: default 값, 인접한 4개 픽셀 값에 거리 가중치 사용
cv2.INTER_NEAREST: 가장 가까운 픽셀 값 사용
cv2.INTER_AREA: 픽셀 영역 관계를 이용한 재샘플링
cv2.INTER_CUBIC: 인정합 16개 픽셀 값에 거리 가중치 사용

borderMode의 값은 아래와 같습니다.
cv2.BORDER_CONSTANT: 고정 색상 값
cv2.BORDER_REPLICATE: 가장자리 복제
cv2.BORDER_WRAP: 반복
cv2.BORDER_REFLECT: 반사

'''


# In[2]:


# 평행 이동 (translate.py)

import cv2
import numpy as np

img = cv2.imread('C:/cdd/moons1.jpg')
rows,cols = img.shape[0:2]  # 영상의 크기

dx, dy = 100, 50            # 이동할 픽셀 거리

# ---① 변환 행렬 생성 
mtrx = np.float32([[1, 0, dx],
                   [0, 1, dy]])  
# ---② 단순 이동
dst = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy))   

# ---③ 탈락된 외곽 픽셀을 파랑색으로 보정
dst2 = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy), None,                         cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (255,0,0) )

# ---④ 탈락된 외곽 픽셀을 원본을 반사 시켜서 보정
dst3 = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy), None,                                 cv2.INTER_LINEAR, cv2.BORDER_REFLECT)

cv2.imshow('original', img)
cv2.imshow('trans',dst)
cv2.imshow('BORDER_CONSTATNT', dst2)
cv2.imshow('BORDER_FEFLECT', dst3)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 이미지 확대 및 축소

# In[ ]:


'''

이미지를 일정 비율로 확대 및 축소하는 방법은 아래와 같습니다. 기존의 좌표에 특정한 값을 곱하면 됩니다.

x_new = a₁ * x_old
y_new = a₂ * y_old

이를 다시 풀어쓰면 아래와 같습니다.

x_new = a₁ * x_old = a₁ * x_old + 0 * y_old + 0 * 1
y_new = a₂ * y_old = 0 * x_old + a₂ * y_old + 0 * 1

변환 행렬은 평행 이동할 때와 마찬가지로 2 x 3 행렬입니다. 
2 x 2 행렬로 나타낼 수 있는데 굳이 2 x 3 행렬로 표현한 이유는 cv2.warpAffine() 함수는 변환 행렬이 2 x 3 행렬이 아니면 
오류를 내기 때문입니다. 기하학적 변환은 이미지 확대 및 축소뿐만 아니라 평행 이동도 있습니다. 
두 변환을 같이 하기 위해 2 x 3 행렬로 맞춘 것입니다.

아래는 변환 행렬을 이용해서 이미지를 확대하고 축소하는 예제 코드입니다.
이미지 평행 이동과 마찬가지로 cv2.warpAffine() 함수를 사용했습니다.

'''


# In[3]:


# 행렬을 이용한 이미지 확대 및 축소 (scale_matrix.py)

import cv2
import numpy as np

img = cv2.imread('C:/cdd/moons1.jpg')
height, width = img.shape[:2]

# --① 0.5배 축소 변환 행렬
m_small = np.float32([[0.5, 0, 0],
                       [0, 0.5,0]])  
# --② 2배 확대 변환 행렬
m_big = np.float32([[2, 0, 0],
                     [0, 2, 0]])  

# --③ 보간법 적용 없이 확대 축소
dst1 = cv2.warpAffine(img, m_small, (int(height*0.5), int(width*0.5)))
dst2 = cv2.warpAffine(img, m_big, (int(height*2), int(width*2)))

# --④ 보간법 적용한 확대 축소
dst3 = cv2.warpAffine(img, m_small, (int(height*0.5), int(width*0.5)),                         None, cv2.INTER_AREA)
dst4 = cv2.warpAffine(img, m_big, (int(height*2), int(width*2)),                         None, cv2.INTER_CUBIC)

# 결과 출력
cv2.imshow("original", img)
cv2.imshow("small", dst1)
cv2.imshow("big", dst2)
cv2.imshow("small INTER_AREA", dst3)
cv2.imshow("big INTER_CUBIC", dst4)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


# cv2.reize()로 이미지 확대 및 축소 (scale_resize.py)

import cv2
import numpy as np

img = cv2.imread('C:/cdd/moons1.jpg')
height, width = img.shape[:2]

#--① 크기 지정으로 축소
#dst1 = cv2.resize(img, (int(width*0.5), int(height*0.5)),\
#                        None, 0, 0, cv2.INTER_AREA)
dst1 = cv2.resize(img, (int(width*0.5), int(height*0.5)),                          interpolation=cv2.INTER_AREA)

#--② 배율 지정으로 확대
dst2 = cv2.resize(img, None,  None, 2, 2, cv2.INTER_CUBIC)
#--③ 결과 출력
cv2.imshow("original", img)
cv2.imshow("small", dst1)
cv2.imshow("big", dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 이미지 회전

# In[13]:


# 변환행렬을 이용한 이미지 회전 (rotate_martix.py)

import cv2
import numpy as np

img = cv2.imread('C:/cdd/moons1.jpg')
rows,cols = img.shape[0:2]

# ---① 라디안 각도 계산(60진법을 호도법으로 변경)
d45 = 45.0 * np.pi / 180    # 45도
d90 = 90.0 * np.pi / 180    # 90도

# ---② 회전을 위한 변환 행렬 생성
m45 = np.float32( [[ np.cos(d45), -1* np.sin(d45), rows//2],
                    [np.sin(d45), np.cos(d45), -1*cols//4]])
m90 = np.float32( [[ np.cos(d90), -1* np.sin(d90), rows],
                    [np.sin(d90), np.cos(d90), 0]])

# ---③ 회전 변환 행렬 적용
r45 = cv2.warpAffine(img,m45,(cols,rows))
r90 = cv2.warpAffine(img,m90,(rows,cols))

# ---④ 결과 출력
cv2.imshow("origin", img)
cv2.imshow("45", r45)
cv2.imshow("90", r90)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[14]:


'''

mtrx = cv2.getRotationMatrix2D(center, angle, scale)
center: 회전축 중심 좌표 (x, y)
angle: 회전할 각도, 60진법
scale: 확대 및 축소비율

'''


# In[15]:


# OpenCv로 회전 변환행렬 구하기 (rotate_getmatrix.py)

import cv2

img = cv2.imread('C:/cdd/moons1.jpg')
rows,cols = img.shape[0:2]

#---① 회전을 위한 변환 행렬 구하기
# 회전축:중앙, 각도:45, 배율:0.5
m45 = cv2.getRotationMatrix2D((cols/2,rows/2),45,0.5) 
# 회전축:중앙, 각도:90, 배율:1.5
m90 = cv2.getRotationMatrix2D((cols/2,rows/2),90,1.5) 

#---② 변환 행렬 적용
img45 = cv2.warpAffine(img, m45,(cols, rows))
img90 = cv2.warpAffine(img, m90,(cols, rows))

#---③ 결과 출력
cv2.imshow('origin',img)
cv2.imshow("45", img45)
cv2.imshow("90", img90)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




