#!/usr/bin/env python
# coding: utf-8

# ### BGR,BGRA

# In[ ]:


# BGR, BGRA, Ahlpha 채널 (rgba.py)

import cv2
import numpy as np

# 기본 값 옵션
img = cv2.imread('C:/cdd/moons1.jpg')   
# IMREAD_COLOR 옵션                   
bgr = cv2.imread('C:/cdd/moons1.jpg', cv2.IMREAD_COLOR)    
# IMREAD_UNCHANGED 옵션
bgra = cv2.imread('C:/cdd/moons1.jpg', cv2.IMREAD_UNCHANGED) 
# 각 옵션에 따른 이미지 shape
print("default", img.shape, "color", bgr.shape, "unchanged", bgra.shape) 

cv2.imshow('bgr', bgr)
cv2.imshow('bgra', bgra)
cv2.imshow('alpha', bgra[:,:,3])  # 알파 채널만 표시
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 색상 이미지를 회색조 이미지로 변환하기

# In[15]:


# BGR 색상 이미지를 회색조 이미지로 변환 (bgr2gray.py)

import cv2
import numpy as np

img = cv2.imread('C:/ccd/moons1.jpg')

img2 = img.astype(np.int16)                # dtype 변경 ---①
b,g,r = cv2.split(img2)                     # 채널 별로 분리 ---②
#b,g,r = img2[:,:,0], img2[:,:,1], img2[:,:,2]
gray1 = ((b + g + r)/3).astype(np.uint8)    # 평균 값 연산후 dtype 변경 ---③

gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # BGR을 그레이 스케일로 변경 ---④
cv2.imshow('original', img)
cv2.imshow('gray1', gray1)
cv2.imshow('gray2', gray2)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ### HSV 방식( H : 색조, S : 채도, V : 명도)

# In[16]:


# BGR을 HSV로 변환 (bgr2hsv.py)

import cv2
import numpy as np

#---① BGR 컬러 스페이스로 원색 픽셀 생성
red_bgr = np.array([[[0,0,255]]], dtype=np.uint8)   # 빨강 값만 갖는 픽셀
green_bgr = np.array([[[0,255,0]]], dtype=np.uint8) # 초록 값만 갖는 픽셀
blue_bgr = np.array([[[255,0,0]]], dtype=np.uint8)  # 파랑 값만 갖는 픽셀
yellow_bgr = np.array([[[0,255,255]]], dtype=np.uint8) # 노랑 값만 갖는 픽셀

#---② BGR 컬러 스페이스를 HSV 컬러 스페이스로 변환
red_hsv = cv2.cvtColor(red_bgr, cv2.COLOR_BGR2HSV);
green_hsv = cv2.cvtColor(green_bgr, cv2.COLOR_BGR2HSV);
blue_hsv = cv2.cvtColor(blue_bgr, cv2.COLOR_BGR2HSV);
yellow_hsv = cv2.cvtColor(yellow_bgr, cv2.COLOR_BGR2HSV);

#---③ HSV로 변환한 픽셀 출력
print("red:",red_hsv)
print("green:", green_hsv)
print("blue", blue_hsv)
print("yellow", yellow_hsv)


# ### YUV, YCbCR 방식

# In[10]:


'''

YUV = YCbCr 

Y : 밝기 
U : 밝기와 파란색과의 색상 차 
V : 밝기와 빨간색의 색상 차

Y(밝기)에는 많은 비트수를 할당하고 U(Cb)와 V(Cr)에는 적은 비트수를 할당하여
데이터를 압축하는 효과를 갖습니다.

'''


# In[17]:


import cv2
import numpy as np

dark = np.array([[[0,0,0]]], dtype = np.uint8)
middle = np.array([[[127,127,127]]], dtype = np.uint8)
bright = np.array([[[255,255,255]]], dtype = np.uint8)

dark_yuv = cv2.cvtColor(dark, cv2.COLOR_BGR2YUV)
middle_yuv = cv2.cvtColor(middle, cv2.COLOR_BGR2YUV)
bright_yuv = cv2.cvtColor(bright, cv2.COLOR_BGR2YUV)

print("dark:", dark_yuv)
print("middle:", middle_yuv)
print("bright", bright_yuv)


# In[ ]:



# 출처 : https://bkshin.tistory.com/category/OpenCV
