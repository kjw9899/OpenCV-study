#!/usr/bin/env python
# coding: utf-8

# ### 스레시홀딩 (바이너리 이미지를 만드는 대표적인 방법)

# In[2]:


#스레시 홀딩이란? 여러 값을 어떤 임계점을 기준으로 두 가지 부류로 나누는 방법을 의미


# ### 전역 스레시홀딩

# In[16]:


import cv2
import numpy as np
import matplotlib.pylab as plt

img = cv2.imread('C:/cdd/232.jpg',cv2.IMREAD_GRAYSCALE) # 이미지를 그레이 스케일로 읽기

# NUMPY API로 바이너리 이미지 만들기
thresh_np = np.zeros_like(img) #img와 동일한 크기의 0으로 채워진 이미지
thresh_np[img > 127] = 255 # 127보다 큰 값만 255로 변경

# OpenCV API로 바이너리 이미지 만들기
ret, thresh_cv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
print(ret)

# 원본과 결과물을 matplotlib.pylab 으로 출력

imgs = {'Original' : img, 'Numpy API' : thresh_np, 'cv2.thresh_cv': thresh_cv}
for i, (key, value) in enumerate(imgs.items()):
    plt.subplot(3,1,i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([]); plt.yticks([])

    plt.show()
    


# In[26]:


# 스레시홀딩 플래그

import cv2
import numpy as np
import matplotlib.pylab as plt

img = cv2.imread('C:/cdd/232.jpg', cv2.IMREAD_GRAYSCALE)

_, t_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
_, t_bininv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
_, t_truc = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
_, t_2zr = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
_, t_2zrinv = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

imgs = {'origin':img, 'BINARY':t_bin, 'BINARY_INV':t_bininv,         'TRUNC':t_truc, 'TOZERO':t_2zr, 'TOZERO_INV':t_2zrinv}
for i, (key, value) in enumerate(imgs.items()):
    plt.subplot(2,3, i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([]);    plt.yticks([])
    
plt.show()
    


# ### 오츠의 이진화 알고리즘

# In[30]:


import cv2
import numpy as np
import matplotlib.pylab as plt

# 이미지를 그레이 스케일로 읽기
img = cv2.imread('C:/cdd/232.jpg', cv2.IMREAD_GRAYSCALE)
# 경계 값을 130으로 지정
_, t_130 = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
#경계 값을 지정하지 않고 OTSU 알고리즘 선택
t, t_otsu = cv2.threshold(img, -1, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU) 
print('otsu threshold :', t) # Otsu 알고리즘으로 선택된 경계 값 출력

imgs = {'Original': img, 't:130':t_130, 'otsu:%d'%t: t_otsu}
for i , (key, value) in enumerate(imgs.items()):
    plt.subplot(1, 3, i+1)
    plt.title(key)
    
    plt.imshow(value, cmap='gray')
    plt.xticks([]); plt.yticks([])

plt.show()


# ### 적응형 스레시홀딩

# In[31]:


import cv2
import numpy as np
import matplotlib.pylab as plt

blk_size = 9 # 블럭 사이즈
C = 5 # 차감 상수
img = cv2.imread('C:/cdd/232.jpg', cv2.IMREAD_GRAYSCALE)

ret, th1 = cv2.threshold(img, 0 , 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,                                      cv2.THRESH_BINARY, blk_size, C)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,                                      cv2.THRESH_BINARY, blk_size, C)

# ---③ 결과를 Matplot으로 출력
imgs = {'Original': img, 'Global-Otsu:%d'%ret:th1,         'Adapted-Mean':th2, 'Adapted-Gaussian': th3}
for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(2,2,i+1)
    plt.title(k)
    plt.imshow(v,'gray')
    plt.xticks([]),plt.yticks([])

plt.show()

# 출처 : https://bkshin.tistory.com/entry/OpenCV-8-%EC%8A%A4%EB%A0%88%EC%8B%9C%ED%99%80%EB%94%A9Thresholding?category=1148027


# In[ ]:




