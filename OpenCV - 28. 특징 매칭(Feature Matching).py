#!/usr/bin/env python
# coding: utf-8

# ### 특징 매칭

# In[ ]:


'''

특징 매칭이란 서로 다은 두 이미지에서 특징점과 특징 디스크립터들을 비교해서 
비슷한 객체끼리 짝짓는 것을 말합니다.

OpenCV는 특징 매칭을 위해 아래와 같은 특징 매칭 인터페이스 함수를 제공합니다.

matcher = cv2.DescriptorMatcher_create(matcherType): 매칭기 생성자
matcherType: 생성할 구현 클래스의 알고리즘
("BruteForce": NORM_L2를 사용하는 BFMatcher,
"BruteForce-L1": NORM_L1을 사용하는 BFMatcher,
"BruteForce-Hamming": NORM_HAMMING을 사용하는 BRMatcher,
"BruteForce-Hamming(2)": NORM_HAMMING2를 사용하는 BFMatcher,
"FlannBased": NORM_L2를 사용하는 FlannBasedMatcher)

'''


# In[1]:


# BFMatcher와 ORB로 매칭

import cv2, numpy as np

img1 = cv2.imread('C:/cdd/bookandme.jpg')
img2 = cv2.imread('C:/cdd/book.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# SIFT 서술자 추출기 생성
detector = cv2.ORB_create()
# 각 영상에 대해 키 포인트와 서술자 추출
kp1, desc1 = detector.detectAndCompute(gray1, None) # 특징점 검출과 특징 디스크립터 계산을 한 번에 수행
kp2, desc2 = detector.detectAndCompute(gray2, None) 

# BFMatcher 생성, Hamming 거리, 상호 체크
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# 매칭 계산 ---④
matches = matcher.match(desc1, desc2)
# 매칭 결과 그리기 ---⑤
res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,                      flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('BFMatcher + ORB', res)
cv2.waitKey()
cv2.destroyAllWindows()


# ### FLANN(Fast Library for Approximate Nearest Neighbors Matching)

# In[2]:


# FLANNMatcher와 ORB로 매칭 (match_flann_orb.py)

import cv2, numpy as np

img1 = cv2.imread('C:/cdd/bookandme.jpg')
img2 = cv2.imread('C:/cdd/book.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ORB 추출기 생성
detector = cv2.ORB_create()
# 키 포인트와 서술자 추출
kp1, desc1 = detector.detectAndCompute(gray1, None)
kp2, desc2 = detector.detectAndCompute(gray2, None)

# 인덱스 파라미터 설정 ---①
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1)
# 검색 파라미터 설정 ---②
search_params=dict(checks=32)
# Flann 매처 생성 ---③
matcher = cv2.FlannBasedMatcher(index_params, search_params)
# 매칭 계산 ---④
matches = matcher.match(desc1, desc2)
# 매칭 그리기
res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,             flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# 결과 출력            
cv2.imshow('Flann + ORB', res)
cv2.waitKey()
cv2.destroyAllWindows()


# In[ ]:


# 출처 : https://bkshin.tistory.com/entry/OpenCV-28-%ED%8A%B9%EC%A7%95-%EB%A7%A4%EC%B9%ADFeature-Matching?category=1148027

