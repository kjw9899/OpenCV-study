#!/usr/bin/env python
# coding: utf-8

# ### 히스토그램

# In[ ]:


'''

cv2.calHist(img, channel, mask, histSize, ranges)
img: 이미지 영상, [img]처럼 리스트로 감싸서 전달
channel: 분석 처리할 채널, 리스트로 감싸서 전달 - 1 채널: [0], 2 채널: [0, 1], 3 채널: [0, 1, 2]
mask: 마스크에 지정한 픽셀만 히스토그램 계산, None이면 전체 영역
histSize: 계급(Bin)의 개수, 채널 개수에 맞게 리스트로 표현 - 1 채널: [256], 2 채널: [256, 256], 3 채널: [256, 256, 256]
ranges: 각 픽셀이 가질 수 있는 값의 범위, RGB인 경우 [0, 256]

'''


# In[5]:


# 회색조 1채널 히스토그램

import cv2
import numpy as np
import matplotlib.pylab as plt

# 이미지 그레이 스케일로 읽기 및 출력 
img = cv2.imread('C:/cdd/mask1.jpg',cv2.IMREAD_GRAYSCALE)
#cv2. imshow('img', img)

#히스토그램 계산 및 그리기
hist = cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(hist)

print("hist.shape:", hist.shape) # 히스토그램의 shape (256,1)
print("hist.sum():", hist.sum(),"img.shape:",img.shape) 
#히스토그램 총 합계와 이미지의 크기
plt.show()
plt.imshow(img)


# ### 색상 이미지 히스토그램

# In[1]:


import cv2
import numpy as np
import matplotlib.pylab as plt

# 이미지 읽기 및 출력

img = cv2.imread('C:/cdd/mask1.jpg')
cv2.imshow('img', img)

# 히스토그램 계산 및 그리기

channels = cv2.split(img)
colors = ('b','r','g')
for (ch, color) in zip (channels, colors):
    hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
    plt.plot(hist, color = color)
plt.show()


# ### 정규화

# In[ ]:


'''

dst = cv2.normalize(src, dst, alpha, beta, type_flag)
src: 정규화 이전의 데이터
dst: 정규화 이후의 데이터
alpha: 정규화 구간 1
beta: 정규화 구간 2, 구간 정규화가 아닌 경우 사용 안 함
type_flag: 정규화 알고리즘 선택 플래그 상수

'''


# In[4]:


# 히스토그램 정규화

import cv2
import numpy as np
import matplotlib.pylab as plt

# 그레이 스케일로 영상 읽기
img = cv2.imread('C:/cdd/232.jpg', cv2.IMREAD_GRAYSCALE)

# 직접 연산한 정규화
img_f = img.astype(np.float32)
img_norm = ((img_f - img_f.min()) * (255) / (img_f.max() - img_f.min()))
img_norm = img_norm.astype(np.uint8)


# OpenCV API를 이용한 정규화
img_norm2 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


# 히스토그램 계산
hist = cv2.calcHist([img], [0], None, [256], [0, 255])
hist_norm = cv2.calcHist([img_norm], [0], None, [256], [0, 255])
hist_norm2 = cv2.calcHist([img_norm2], [0], None, [256], [0, 255])

# cv2.imshow('Before', img)
# cv2.imshow('Manual', img_norm)
# cv2.imshow('cv2.normalize()', img_norm2)


hists = {'Before' : hist, 'Manual':hist_norm, 'cv2.normalize()':hist_norm2}
for i, (k, v) in enumerate(hists.items()):
    plt.subplot(1,3,i+1)
    plt.title(k)
    plt.plot(v)
plt.show()


# ### 평탄화

# In[ ]:


'''

dst = cv2.equalizeHist(src, dst)
src: 대상 이미지, 8비트 1 채널
dst(optional): 결과 이미지

'''


# In[5]:


# 회색조 이미지에 평탄화 적용 (histo_equalize.py)

import cv2
import numpy as np
import matplotlib.pylab as plt

#--① 대상 영상으로 그레이 스케일로 읽기
img = cv2.imread('C:/cdd/mask1.jpg', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape[:2]

#--② 이퀄라이즈 연산을 직접 적용
hist = cv2.calcHist([img], [0], None, [256], [0, 256]) #히스토그램 계산
cdf = hist.cumsum()                                     # 누적 히스토그램 
cdf_m = np.ma.masked_equal(cdf, 0)                      # 0(zero)인 값을 NaN으로 제거
cdf_m = (cdf_m - cdf_m.min()) /(rows * cols) * 255      # 이퀄라이즈 히스토그램 계산
cdf = np.ma.filled(cdf_m,0).astype('uint8')             # NaN을 다시 0으로 환원
print(cdf.shape)
img2 = cdf[img]                                         # 히스토그램을 픽셀로 맵핑

#--③ OpenCV API로 이퀄라이즈 히스토그램 적용
img3 = cv2.equalizeHist(img)

#--④ 이퀄라이즈 결과 히스토그램 계산
hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
hist3 = cv2.calcHist([img3], [0], None, [256], [0, 256])

#--⑤ 결과 출력
cv2.imshow('Before', img)
cv2.imshow('Manual', img2)
cv2.imshow('cv2.equalizeHist()', img3)
hists = {'Before':hist, 'Manual':hist2, 'cv2.equalizeHist()':hist3}
for i, (k, v) in enumerate(hists.items()):
    plt.subplot(1,3,i+1)
    plt.title(k)
    plt.plot(v)
plt.show()


# In[6]:


#색상 이미지에 대한 평탄화 적용 (histo_equalize_yuv.py)

import numpy as np, cv2

img = cv2.imread('C:/cdd/mask1.jpg') #이미지 읽기, BGR 스케일

#--① 컬러 스케일을 BGR에서 YUV로 변경
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) 

#--② YUV 컬러 스케일의 첫번째 채널에 대해서 이퀄라이즈 적용
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0]) 

#--③ 컬러 스케일을 YUV에서 BGR로 변경
img2 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) 

cv2.imshow('Before', img)
cv2.imshow('After', img2)
cv2.waitKey()
cv2.destroyAllWindows()


# ### CLAHE (Contrast Limited Adaptive Histogram Equalization)

# In[ ]:


'''

clahe = cv2.createCLAHE(clipLimit, tileGridSize)
clipLimit: 대비(Contrast) 제한 경계 값, default=40.0
tileGridSize: 영역 크기, default=8 x 8
clahe: 생성된 CLAHE 객체

clahe.apply(src): CLAHE 적용
src: 입력 이미지

'''


# In[10]:


# CLAHE 적용 (histo_clahe.py)

import cv2
import numpy as np
import matplotlib.pylab as plt

#--①이미지 읽어서 YUV 컬러스페이스로 변경
img = cv2.imread('C:/cdd/smile2.jpg')
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# 이미지 읽어서 YUV 컬러스페이스로 변경

img_eq = img_yuv.copy()
clahe = cv2.equalizeHist(img_eq[:,:,0])
img_eq = cv2.cvtColor(img_eq, cv2.COLOR_YUV2BGR)

#--③ 밝기 채널에 대해서 CLAHE 적용
img_clahe = img_yuv.copy()
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) #CLAHE 생성
img_clahe[:,:,0] = clahe.apply(img_clahe[:,:,0])           #CLAHE 적용
img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_YUV2BGR)

#--④ 결과 출력
cv2.imshow('Before', img)
cv2.imshow('CLAHE', img_clahe)
cv2.imshow('equalizeHist', img_eq)
cv2.waitKey()
cv2.destroyAllWindows()


# In[ ]:




# 출처 : https://bkshin.tistory.com/category/OpenCV
