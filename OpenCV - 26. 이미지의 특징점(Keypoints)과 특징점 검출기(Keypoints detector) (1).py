#!/usr/bin/env python
# coding: utf-8

# ### 이미지 특징점

# In[1]:


'''

이미지 특징점이란 말 그대로 이미지에서 특징이 되는 부분을 의미합니다.
이미지끼리 서로 매칭이 되는지 확인을 할 때 각 이미지에서의 특징이 되는 부분끼리 비교를 합니다.
즉, 이미지 매칭 시 사용되는 것이 바로 특징점입니다.
특징점은 영어로 키 포인트(Keypoint)라고도 합니다.
보통 특징점이 되는 부분은 물테의 모서리나 코너입니다.
그래서 대부분의 특징점 검출을 코너 검출을 바탕으로 하고 있습니다.


'''


# ### 해리스 코너 검출

# In[ ]:


'''

사각형을 사각형이라고 인지할 수 있는 건 4개의 꼭짓점이 있기 때문입니다.
삼각형도 3개의 꼭짓점이 있기 때문에 삼각형이라고 인지할 수 있습니다.
마찬가지로 우리가 어떤 물체를 볼 때 꼭짓점을 더 유심히 보는 경향이 있습니다.
즉 물체를 인식할 때 물체의 코너 부분에 관심을 둡니다.
이미지 상의 코너를 잘 찾아낸다면 물체를 보다 쉽게인식할 수 잇을 것입니다.

코너를 검출하기 위한 방법으로는 해리스 코너 검출이 있습니다.
해리스 코너 검출은 소벨 미분으로 경곗값을 검출하면서 경계값의 경사도 변화량을 측정하여 변화량이
수직, 수평을 대각선 방향으로 크게 변화하는 것을 코너로 판단합니다.

dst = cv2.cornerHarris(src, blockSize, ksize, k, dst, borderType)
src: 입력 이미지, 그레이 스케일
blockSize: 이웃 픽셀 범위
ksize: 소벨 미분 필터 크기
k(optional): 코너 검출 상수 (보토 0.04~0.06)
dst(optional): 코너 검출 결과 (src와 같은 크기의 1 채널 배열, 변화량의 값, 지역 최대 값이 코너점을 의미)
borderType(optional): 외곽 영역 보정 형식


'''


# In[9]:


# 해리스 코너 검출 (corner_harris.py)

import cv2
import numpy as np

img = cv2.imread('C:/cdd/house.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 해리스 코너 검출 ---①
corner = cv2.cornerHarris(gray, 2, 3, 0.04)
# 변화량 결과의 최대값 10% 이상의 좌표 구하기 ---②
coord = np.where(corner > 0.1* corner.max())
coord = np.stack((coord[1], coord[0]), axis=-1)

# 코너 좌표에 동그리미 그리기 ---③
for x, y in coord:
    cv2.circle(img, (x,y), 5, (0,0,255), 1, cv2.LINE_AA)

# 변화량을 영상으로 표현하기 위해서 0~255로 정규화 ---④
corner_norm = cv2.normalize(corner, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
# 화면에 출력
corner_norm = cv2.cvtColor(corner_norm, cv2.COLOR_GRAY2BGR)
merged = np.hstack((corner_norm, img))
cv2.imshow('Harris Corner', merged)
cv2.waitKey()
cv2.destroyAllWindows()


# ### 시 - 토마시 검출

# In[2]:


# 시와 토마시 코너 검출

import cv2
import numpy as np

img = cv2.imread('C:/cdd/house.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 시 - 토마스의 코너 검출 메서드
corners = cv2.goodFeaturesToTrack(gray, 80, 0.01, 10)
# 실수 좌표를 정수 좌표로 변환
corners = np.int32(corners)

for corner in corners:
    x, y = corner[0]
    cv2.circle(img, (x, y), 5, (0,0,255), 1, cv2.LINE_AA)

# 좌표에 동그라미 표시
cv2.imshow('Corners', img)
cv2.waitKey()
cv2.destroyAllWindows()


# ### 특징점 검출기

# In[ ]:


'''

keypoints = detector.detect(img, mask): 특징점 검출 함수
img: 입력 이미지
mask(optional): 검출 제외 마스크
keypoints: 특징점 검출 결과 (KeyPoint의 리스트)

Keypoint: 특징점 정보를 담는 객체
pt: 특징점 좌표(x, y), float 타입으로 정수 변환 필요
size: 의미 있는 특징점 이웃의 반지름
angle: 특징점 방향 (시계방향, -1=의미 없음)
response: 특징점 반응 강도 (추출기에 따라 다름)
octave: 발견된 이미지 피라미드 계층
class_id: 특징점이 속한 객체 ID

outImg = cv2.drawKeypoints(img, keypoints, outImg, color, flags)
img: 입력 이미지
keypoints: 표시할 특징점 리스트
outImg: 특징점이 그려진 결과 이미지
color(optional): 표시할 색상 (default: 랜덤)
flags(optional): 표시 방법 (cv2.DRAW_MATCHES_FLAGS_DEFAULT: 좌표 중심에 동그라미만 그림(default), 
cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS: 동그라미의 크기를 size와 angle을 반영해서 그림)
'''


# ### GFTTDetector

# In[ ]:


'''

detector = cv2.GFTTDetector_create(img, maxCorners, qualityLevel, minDistance, corners, mask, blockSize, useHarrisDetector, k)
모든 파라미터는 cv2.goodFeaturesToTrack()과 동일



corners = cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance, corners, mask, blockSize, useHarrisDetector, k)
img: 입력 이미지
maxCorners: 얻고 싶은 코너의 개수, 강한 것 순으로
qualityLevel: 코너로 판단할 스레시홀드 값
minDistance: 코너 간 최소 거리
mask(optional): 검출에 제외할 마스크
blockSize(optional)=3: 코너 주변 영역의 크기
useHarrisDetector(optional)=False: 코너 검출 방법 선택 (True: 해리스 코너 검출 방법, False: 시와 토마시 코너 검출 방법)
k(optional): 해리스 코너 검출 방법에 사용할 k 계수
corners: 코너 검출 좌표 결과, N x 1 x 2 크기의 배열, 실수 값이므로 정수로 변형 필요

'''


# In[4]:


import cv2
import numpy as np

img = cv2.imread('C:/cdd/house.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Good feature to trac 검출기 생성
gftt = cv2.GFTTDetector_create()
# 특징점 검출
keypoints = gftt.detect(gray, None)
# 특징점 그리기 ---③
img_draw = cv2.drawKeypoints(img, keypoints, None)

# 결과 출력 ---④
cv2.imshow('GFTTDectector', img_draw)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### FAST(Feature from Accelerated Segment Test)

# In[ ]:


'''

픽셀을 중심으로 특정 개수의 픽셀릏 중심으로 특정 개수의 픽셀로 원을 그려서 그 안의 픽셀들이 중심 픽셀 값보다
임계 값 이상 밝거나 어두운 것이 일정 개수 이상 연속되면 코너로 판단합니다.
 다시 말해 어떤 점 p가 특징점인지 여부를 판단할 때, p를 중심으로 하는 원 상이 16개 픽셀 값을 봅니다.
 p보다 임계 값 이상 밝은 픽셀들이 n개 이상 연속되어 있거나 또는 임계 값 이상 어두운 픽셀들이 n개 이상
 연속되어 있으면 p를 특징점이라고 판단합니다.
 
detector = cv2.FastFeatureDetector_create(threshold, nonmaxSuppression, type)
threshold(optional): 코너 판단 임계 값 (default=10)
nonmaxSuppression(optional): 최대 점수가 아닌 코너 억제 (default=True)
type(optional): 엣지 검출 패턴 (cv2.FastFeatureDetector_TYPE_9_16: 16개 중 9개 연속(default),
cv2.FastFeatureDetector_TYPE_7_12: 12개 중 7개 연속, cv2.FastFeatureDetector_TYPE_5_8: 8개 중 5개 연속)
'''


# In[1]:


# FAST로 특징점 검출 (kpt_fast.py)

import cv2
import numpy as np

img = cv2.imread('C:/cdd/house.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# FASt 특징 검출기 생성 ---①
fast = cv2.FastFeatureDetector_create(50)
# 특징점 검출 ---②
keypoints = fast.detect(gray, None)
# 특징점 그리기 ---③
img = cv2.drawKeypoints(img, keypoints, None)
# 결과 출력 ---④
cv2.imshow('FAST', img)
cv2.waitKey()
cv2.destroyAllWindows()


# In[ ]:


'''

BLOB(Binary Large Object)는 이진 스케일로 연결된 픽셀 그룹을 말합니다. 
SimpleBlobDetector는 자잘한 객체는 노이즈로 여기고 특정 크기 이상의 큰 객체만 찾아내는 검출기입니다. 
SimpleBlobDetector는 아래와 같이 생성합니다.

detector = cv2.SimpleBlobDetector_create([parameters]): BLOB 검출기 생성자
parametes는 다음과 같습니다.

cv2.SimpleBlobDetector_Params()
minThreshold, maxThreshold, thresholdStep: BLOB를 생성하기 위한 경계 값
(minThreshold에서 maxThreshold를 넘지 않을 때까지 thresholdStep만큼 증가)
minRepeatability: BLOB에 참여하기 위한 연속된 경계 값의 개수
minDistBetweenBlobs: 두 BLOB을 하나의 BLOB으로 간주하는 거리
filterByArea: 면적 필터 옵션
minArea, maxArea: min~max 범위의 면적만 BLOB으로 검출
filterByCircularity: 원형 비율 필터 옵션
minCircularity, maxCircularity: min~max 범위의 원형 비율만 BLOB으로 검출
filterByColor: 밝기를 이용한 필터 옵션
blobColor: 0 = 검은색 BLOB 검출, 255 = 흰색 BLOB 검출
filterByConvexity: 볼록 비율 필터 옵션
minConvexity, maxConvexity: min~max 범위의 볼록 비율만 BLOB으로 검출
filterByInertia: 관성 비율 필터 옵션
minInertiaRatio, maxInertiaRatio: min~max 범위의 관성 비율만 BLOB으로 검출

'''


# In[2]:


# SimpleBolbDetector 검출기 (kpt_blob.py)

import cv2
import numpy as np
 
img = cv2.imread("C:/cdd/house.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# SimpleBlobDetector 생성 ---①
detector = cv2.SimpleBlobDetector_create()
# 키 포인트 검출 ---②
keypoints = detector.detect(gray)
# 키 포인트를 빨간색으로 표시 ---③
img = cv2.drawKeypoints(img, keypoints, None, (0,0,255),                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
cv2.imshow("Blob", img)
cv2.waitKey(0)


# In[3]:


# 필터 옵션으로 생성한 SimpleBlobDetector 검출기 (kpt_blob_param.py)

import cv2
import numpy as np
 
img = cv2.imread("C:/cdd/house.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# blob 검출 필터 파라미터 생성 ---①
params = cv2.SimpleBlobDetector_Params()

# 경계값 조정 ---②
params.minThreshold = 10
params.maxThreshold = 240
params.thresholdStep = 5
# 면적 필터 켜고 최소 값 지정 ---③
params.filterByArea = True
params.minArea = 200
  
# 컬러, 볼록 비율, 원형비율 필터 옵션 끄기 ---④
params.filterByColor = False
params.filterByConvexity = False
params.filterByInertia = False
params.filterByCircularity = False 

# 필터 파라미터로 blob 검출기 생성 ---⑤
detector = cv2.SimpleBlobDetector_create(params)
# 키 포인트 검출 ---⑥
keypoints = detector.detect(gray)
# 키 포인트 그리기 ---⑦
img_draw = cv2.drawKeypoints(img, keypoints, None, None,                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# 결과 출력 ---⑧
cv2.imshow("Blob with Params", img_draw)
cv2.waitKey(0)


# In[ ]:


# 출처 :  https://bkshin.tistory.com/entry/OpenCV-26-%EC%9D%B4%EB%AF%B8%EC%A7%80%EC%9D%98-%ED%8A%B9%EC%A7%95%EA%B3%BC-%ED%82%A4-%ED%8F%AC%EC%9D%B8%ED%8A%B8?category=1148027

