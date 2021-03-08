#!/usr/bin/env python
# coding: utf-8

# ### 가우시안 피라미드

# In[ ]:


'''

dst = cv2.pyrDown(src, dst, dstsize, borderType)
dst = cv2.pyrUp(src, dst, dstsize, borderType)
src: 입력 영상
dst: 결과 영상
distsize: 결과 영상 크기
borderType: 외곽 보정 방식

'''


# In[ ]:


# 가우시안 이미지 피라미드

import cv2

img = cv2.imread('C:/cdd/moons1.jpg')

# 가우시안 이미지 피라미드 축소
smaller = cv2.pyrDown(img) # img *1/4
# 가우시안 이미지 피라미드 확대
bigger = cv2.pyrUp(img) # img * 4

# 결과 출력
cv2.imshow('img', img)
cv2.imshow('pyrDown', smaller)
cv2.imshow('pyrUp', bigger)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[1]:


# 라플라시안 피라미드로 영상 복원 (pyramid_laplacian.py)

import cv2
import numpy as np

img = cv2.imread('C:/cdd/232.jpg')

# 원본 영상을 가우시안 피라미드로 축소
smaller = cv2.pyrDown(img)
# 축소한 영상을 가우시안 피라미드로 확대
bigger = cv2.pyrUp(smaller)

# 원본에서 확대한 영상 빼기
laplacian = cv2.subtract(img, bigger)
# 확대 한 영상에 라플라시안 영상 더해서 복원
restored = bigger + laplacian

# 결과 출력 (원본 영상, 라플라시안, 확대 영상, 복원 영상)
merged = np.hstack((img, laplacian, bigger, restored))
cv2.imshow('Laplacian Pyramid', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




