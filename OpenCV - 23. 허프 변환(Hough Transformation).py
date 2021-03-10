#!/usr/bin/env python
# coding: utf-8

# ### 허프 변환

# In[ ]:


'''

허프 변환을 활용해 이미지에서 직선이나 원과 같은 다양한 모양을 인식할 수 있습니다. 

어떤 점이 주어졋다면, 그 점을 기준으로 그 점을 지나는 모든 직선의 방정식의 파라미터를 저장해놓고
여러개의 점들에 대해 반복하여 파라미터의 빈도수를 누적함으로서, 
같은 직선에 속하는 점들이 몇개나 있는지 검사하는 방법이다.
'''


# ### 허프 선 변환

# In[ ]:


'''

이미지는 수많은 픽셀로 구성되어 있습니다. 
그 픽셀 중 서로 직선 관계를 갖는 픽셀들만 골라내는 것이 허프 선 변환의 핵심입니다. 


lines = cv2.HoughLines(img, rho, theta, threshold, lines, srn=0, stn=0, min_theta, max_theta)
img: 입력 이미지, 1 채널 바이너리 스케일
rho: 거리 측정 해상도, 0~1
theta: 각도, 라디안 단위 (np.pi/0~180)
threshold: 직선으로 판단할 최소한의 동일 개수 (작은 값: 정확도 감소, 검출 개수 증가 / 큰 값: 정확도 증가, 검출 개수 감소)
lines: 검출 결과, N x 1 x 2 배열 (r, Θ)
srn, stn: 멀티 스케일 허프 변환에 사용, 선 검출에서는 사용 안 함
min_theta, max_theta: 검출을 위해 사용할 최대, 최소 각도

'''


# In[3]:


# 허프 선 검출

import cv2
import numpy as np

img = cv2.imread('C:/cdd/badook.jpg')
img2 = img.copy()
h, w = img.shape[:2]

# 그레이 스케일 변환 및 엣지 검출
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(imgray, 100, 200)

# 허프 선 검출, 직선으로 판단할 최소한의 점은 130개로 지정
lines =cv2.HoughLines(edges, 1, np.pi/180, 130)
for line in lines : # 검출된 모든 선 순회
    r, theta = line[0] # 거리와 각도
    tx, ty = np.cos(theta), np.sin(theta) # x,y 축에 대한 삼각비
    x0, y0 = tx*r, ty*r  #x, y 기준(절편) 좌표
    # 기준 좌표에 빨강색 점 그리기
    cv2.circle(img2, (abs(x0), abs(y0)), 3, (0,0,255), -1)
    # 직선 방정식으로 그리기 위한 시작점, 끝점 계산
    x1, y1 = int(x0 + w*(-ty)), int(y0 + h * tx)
    x2, y2 = int(x0 - w*(-ty)), int(y0 - h * tx)
    # 선그리기
    cv2.line(img2, (x1, y1), (x2, y2), (0,255,0), 1)

#결과 출력    
merged = np.hstack((img, img2))
cv2.imshow('hough line', merged)
cv2.waitKey()
cv2.destroyAllWindows()


# ### 확률적 허프 선 변환

# In[ ]:


'''

허프 선 검출은 모든 점에 대해 수많은 선을 그어서 직선을 찾기 때문에 견산량이 무척 많습니다.
이를 개선하기 위한 방법이 확률적 허프 선 변환입니다. 
이는 모든 점을 고려하지 않고 무작위로 선정한 픽셀에 대해 허프 변환을 수행하고 점차 그 수를 증가시키는 방법입니다.

lines = cv2.HoughLinesP(img, rho, theta, threshold, lines, minLineLength, maxLineGap)
minLineLength(optional): 선으로 인정할 최소 길이
maxLineGap(optional): 선으로 판단할 최대 간격
lines: 검출된 선 좌표, N x 1 x 4 배열 (x1, y1, x2, y2)
이외의 파라미터는 cv2.HoughLines()와 동일

'''


# In[10]:


# 확률적 허프 변환으로 선 검출 

import cv2
import numpy as np

img = cv2.imread('C:/cdd/1234.jpg')
img2 = img.copy()

# 그레이 스케일로 변환 및 엣지 검출 
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(imgray, 50, 200)

# 확률 허프 변환 적용
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, None, 20, 2 )
for line in lines :
    # 검출된 선 그리기
    x1, y1, x2, y2 = line[0]
    cv2.line(img2, (x1,y1), (x2,y2), (0,255,0),1 )
    
merged = np.hstack((img, img2))
cv2.imshow('Probability hough line', merged)
cv2.waitKey()
cv2.destroyAllWindows()


# ### 허프 원 변환

# In[ ]:


'''

circle = cv2.HoughCircles(img, method, dp, minDist, circles, param1, param2, minRadius, maxRadius)
img: 입력 이미지, 1채널 배열
method: 검출 방식 선택 (현재 cv2.HOUGH_GRADIENT만 가능)
dp: 입력 영상과 경사 누적의 해상도 반비례율, 1: 입력과 동일, 값이 커질수록 부정확
minDist: 원들 중심 간의 최소 거리 (0: 에러, 0이면 동심원이 검출 불가하므로)
circles(optional): 검출 원 결과, N x 1 x 3 부동 소수점 배열 (x, y, 반지름)
param1(optional): 캐니 엣지에 전달할 스레시홀드 최대 값 (최소 값은 최대 값의 2배 작은 값을 전달)
param2(optional): 경사도 누적 경계 값 (값이 작을수록 잘못된 원 검출)
minRadius, maxRadius(optional): 원의 최소 반지름, 최대 반지름 (0이면 이미지 전체의 크기)

'''


# In[5]:


# 허프 원 검출 (hough_circle.py)

import cv2
import numpy as np

img = cv2.imread('C:/cdd/1234.jpg')
# 그레이 스케일 변환 ---①
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 노이즈 제거를 위한 가우시안 블러 ---②
blur = cv2.GaussianBlur(gray, (3,3), 0)
# 허프 원 변환 적용( dp=1.2, minDist=30, cany_max=200 ) ---③
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.2, 30, None, 200)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # 원 둘레에 초록색 원 그리기
        cv2.circle(img,(i[0], i[1]), i[2], (0, 255, 0), 2)
        # 원 중심점에 빨강색 원 그리기
        cv2.circle(img, (i[0], i[1]), 2, (0,0,255), 5)

# 결과 출력
cv2.imshow('hough circle', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


# 출처 : http://localhost:8888/notebooks/kim/OpenCV/OpenCV%20-%2023.%20%ED%97%88%ED%94%84%20%EB%B3%80%ED%99%98(Hough%20Transformation).ipynb#%ED%97%88%ED%94%84-%EC%9B%90-%EB%B3%80%ED%99%98

