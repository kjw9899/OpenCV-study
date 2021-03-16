#!/usr/bin/env python
# coding: utf-8

# ### 특징 디스크립터

# In[ ]:


'''

이 특징점은 객체의 좌표 뿐만아니라 그 주변 픽셀과의 관계에 대한 정보를 가집니다. 
그중 가장 대표적인 것이 size와 angle 속성이며, 코너(corner)점인 경우 코너의 경사도와 방향도 속성으로 가집니다.
특징 디스크립터란 특징점 주변 픽셀을 일정한 크기의 블록으로 나누어 각 블록에 속한 픽셀의
그레디언트 히스토그램을 계산한 것입니다. 주로 특징점 주변의 밝기, 색상, 방향, 크기 등의 정보가 
포함되어 있습니다.
추출하는 알고리즘에 따라 특징 디스크립터가 일부 달라질 수는 있습니다. 일반적으로 특징점 주변의 블록 크기에 
8방향(상,하,좌,우 및 네 방향의 대각선) 경사도를 표현하는 경우가 많습니다. 
4 * 4 크기의 블록인 경우 한 개의 특징점 당 4 * 4 * 8 =128개의 값을 갖습니다.

OpenCV는 특징 디스크립터를 추출하기 위해 다음과 같은 함수를 제공합니다.

keypoints, descriptors = detector.compute(image, keypoins, descriptors): 특징점을 전달하면 특징 디스크립터를 계산해서 반환
keypoints, descriptors = detector.detectAndCompute(image, mask, decriptors, useProvidedKeypoints)
: 특징점 검출과 특징 디스크립터 계산을 한 번에 수행
image: 입력 이미지
keypoints: 디스크립터 계산을 위해 사용할 특징점
descriptors(optional): 계산된 디스크립터
mask(optional): 특징점 검출에 사용할 마스크
useProvidedKeypoints(optional): True인 경우 특징점 검출을 수행하지 않음

keypoints, descriptors = detector.compute(image, keypoins, descriptors): 특징점을 전달하면 특징 디스크립터를 계산해서 반환
keypoints, descriptors = detector.detectAndCompute(image, mask, decriptors, useProvidedKeypoints): 특징점 검출과 특징 디스크립터 계산을 한 번에 수행
image: 입력 이미지
keypoints: 디스크립터 계산을 위해 사용할 특징점
descriptors(optional): 계산된 디스크립터
mask(optional): 특징점 검출에 사용할 마스크
useProvidedKeypoints(optional): True인 경우 특징점 검출을 수행하지 않음


'''


# ### ORB (Oriented and Rotated BRIEF)

# In[2]:


'''

디스크립터 검출기 중 BRIEF(Binary Robust Independent Elementary Features)라는 것이 있습니다.
BRIEF는 특징점 검출은 지원하지 않는 디스크립터 추출기입니다.
이 BRIEF에 방향과 회전을 고려하도록 개선한 알고리즘이 바로  ORB입니다.
이 알고리즘은 특징점 검출 알고리즘으로 FAST를 사용하고 회전과 방향을 고려하도록 개선했으며 
속도도 빨라 SIFT와 SURF의 좋은 대안으로 사용됩니다. ORB 객체생성은 다음과 같이 합니다.

detector = cv2.ORB_create(nfeatures, scaleFactor, nlevels, edgeThreshold, 
firstLevel, WTA_K, scoreType, patchSize, fastThreshold)

nfeatures(optional): 검출할 최대 특징 수 (default=500)
scaleFactor(optional): 이미지 피라미드 비율 (default=1.2)
nlevels(optional): 이미지 피라미드 계층 수 (default=8)
edgeThreshold(optional): 검색에서 제외할 테두리 크기, patchSize와 맞출 것 (default=31)
firstLevel(optional): 최초 이미지 피라미드 계층 단계 (default=0)
WTA_K(optional): 임의 좌표 생성 수 (default=2)
scoreType(optional): 특징점 검출에 사용할 방식 (cv2.ORB_HARRIS_SCORE: 해리스 코너 검출(default), cv2.ORB_FAST_SCORE: FAST 코너 검출)
patchSize(optional): 디스크립터의 패치 크기 (default=31)
fastThreshold(optional): FAST에 사용할 임계 값 (default=20)

'''


# In[6]:


# ORB로 특징점 및 특징 디스크립터 검출

import cv2
import numpy as np

img = cv2.imread('C:/cdd/house.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ORB 추출기 생성
orb = cv2.ORB_create()
# 키 포인트 검출과 서술자 계산
keypoints, dscriptor = orb.detectAndCompute(img, None)
# 키 포인트 그리기
img_draw = cv2.drawKeypoints(img, keypoints, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 결과 출력
cv2.imshow('ORB', img_draw)
cv2.waitKey()
cv2.destroyAllWindows()


# In[4]:


# 출처 : https://bkshin.tistory.com/entry/OpenCV-27-%ED%8A%B9%EC%A7%95-%EB%94%94%EC%8A%A4%ED%81%AC%EB%A6%BD%ED%84%B0-%EA%B2%80%EC%B6%9C%EA%B8%B0-SIFT-SURF-ORB?category=1148027


# In[5]:





# In[ ]:




