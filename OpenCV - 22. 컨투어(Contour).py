#!/usr/bin/env python
# coding: utf-8

# ### 컨투어

# In[1]:


'''

컨투어는 등고선을 의미합니다. 등고선은 지형의 높이가 같은 영역을 하나의 선으로 표시한 것입니다.
영상에서 컨투어를 그리면 모양을 쉽게 인식할 수 있습니다.

'''


# In[ ]:


'''

dst, contours, hierarchy = cv2.findContours(src, mode, method, contours, hierarchy, offset)
src: 입력 영상, 검정과 흰색으로 구성된 바이너리 이미지
mode: 컨투어 제공 방식 (cv2.RETR_EXTERNAL: 가장 바깥쪽 라인만 생성, cv2.RETR_LIST: 모든 라인을 계층 없이 생성, cv2.RET_CCOMP: 모든 라인을 2 계층으로 생성, cv2.RETR_TREE: 모든 라인의 모든 계층 정보를 트리 구조로 생성)
method: 근사 값 방식 (cv2.CHAIN_APPROX_NONE: 근사 없이 모든 좌표 제공, cv2.CHAIN_APPROX_SIMPLE: 컨투어 꼭짓점 좌표만 제공, cv2.CHAIN_APPROX_TC89_L1: Teh-Chin 알고리즘으로 좌표 개수 축소, cv2.CHAIN_APPROX_TC89_KC0S: Teh-Chin 알고리즘으로 좌표 개수 축소)
contours(optional): 검출한 컨투어 좌표 (list type)
hierarchy(optional): 컨투어 계층 정보 (Next, Prev, FirstChild, Parent, -1 [해당 없음])
offset(optional): ROI 등으로 인해 이동한 컨투어 좌표의 오프셋


'''


# In[2]:


# 컨투어 계층 트리 (cntr_hierachy.py)

import cv2
import numpy as np

# 영상 읽기
img = cv2.imread('C:/cdd/223.jpg')
img2 = img.copy()
# 바이너리 이미지로 변환
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, imthres = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)

# 가장 바깥 컨투어만 수집   --- ①
im2, contour, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL,                                                 cv2.CHAIN_APPROX_NONE)
# 컨투어 갯수와 계층 트리 출력 --- ②
print(len(contour), hierarchy)

# 모든 컨투어를 트리 계층 으로 수집 ---③
im2, contour2, hierarchy = cv2.findContours(imthres, cv2.RETR_TREE,                                             cv2.CHAIN_APPROX_SIMPLE)
# 컨투어 갯수와 계층 트리 출력 ---④
print(len(contour2), hierarchy)

# 가장 바깥 컨투어만 그리기 ---⑤
cv2.drawContours(img, contour, -1, (0,255,0), 3)
# 모든 컨투어 그리기 ---⑥
for idx, cont in enumerate(contour2): 
    # 랜덤한 컬러 추출 ---⑦
    color = [int(i) for i in np.random.randint(0,255, 3)]
    # 컨투어 인덱스 마다 랜덤한 색상으로 그리기 ---⑧
    cv2.drawContours(img2, contour2, idx, color, 3)
    # 컨투어 첫 좌표에 인덱스 숫자 표시 ---⑨
    cv2.putText(img2, str(idx), tuple(cont[0][0]), cv2.FONT_HERSHEY_PLAIN,                                                             1, (0,0,255))

# 화면 출력
cv2.imshow('RETR_EXTERNAL', img)
cv2.imshow('RETR_TREE', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[5]:


import cv2 as cv

img_color = cv.imread('C:/cdd/shapes.jpg')
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
ret, img_binary = cv.threshold(img_gray, 127, 255, 0)
contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

cv.drawContours(img_color, contours, 0, (0, 255, 0), 3) 
cv.drawContours(img_color, contours, 1, (255, 0, 0), 3) 


cv.imshow("result", img_color)
cv.waitKey(0)
cv.destroyAllWindows()


# In[4]:


# 컨투어 계층 트리 (cntr_hierachy.py)

import cv2
import numpy as np

# 영상 읽기
img = cv2.imread('C:/cdd/shapes.jpg')
img2 = img.copy()
# 바이너리 이미지로 변환
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, imthres = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)

# 가장 바깥 컨투어만 수집   --- ①
contour, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL,                                                 cv2.CHAIN_APPROX_NONE)
# 컨투어 갯수와 계층 트리 출력 --- ②
print(len(contour), hierarchy)

# 모든 컨투어를 트리 계층 으로 수집 ---③
contour2, hierarchy = cv2.findContours(imthres, cv2.RETR_TREE,                                             cv2.CHAIN_APPROX_SIMPLE)
# 컨투어 갯수와 계층 트리 출력 ---④
print(len(contour2), hierarchy)

# 가장 바깥 컨투어만 그리기 ---⑤
cv2.drawContours(img, contour, -1, (0,255,0), 3)
# 모든 컨투어 그리기 ---⑥
for idx, cont in enumerate(contour2): 
    # 랜덤한 컬러 추출 ---⑦
    color = [int(i) for i in np.random.randint(0,255, 3)]
    # 컨투어 인덱스 마다 랜덤한 색상으로 그리기 ---⑧
    cv2.drawContours(img2, contour2, idx, color, 3)
    # 컨투어 첫 좌표에 인덱스 숫자 표시 ---⑨
    cv2.putText(img2, str(idx), tuple(cont[0][0]), cv2.FONT_HERSHEY_PLAIN,                                                             1, (0,0,255))

# 화면 출력
cv2.imshow('RETR_EXTERNAL', img)
cv2.imshow('RETR_TREE', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 컨투어를 감싸는 도형 그리기

# In[ ]:


'''

OpenCV를 활용하면 컨투어를 감싸는 도형을 그릴 수도 있습니다. 
컨투어를 감싸는 도형을 그리는 아래 함수들에 대해 먼저 알아보겠습니다.

x, y, w, h = cv2.boundingRect(contour): 좌표를 감싸는 사각형 반환
x, y: 사각형의 왼쪽 상단 좌표
w, h: 사각형의 폭과 높이

rotateRect = cv2.minAreaRect(contour): 좌표를 감싸는 최소한의 사각형 계산

vertex = cv2.boxPoints(rotateRect): rotateRect로부터 꼭짓점 좌표 계산
vertex: 4개의 꼭짓점 좌표, 소수점 포함이므로 정수 변환 필요

center, radius = cv2.minEnclosingCircle(contour): 좌표를 감싸는 최소한의 동그라미 계산
center: 원점 좌표(x, y)
radius: 반지름

area, triangle = cv2.minEnclosingTriangle(points): 좌표를 감싸는 최소한의 삼각형 게산
area: 넓이

triangle: 3개의 꼭짓점 좌표

ellipse = cv2.fitEllipse(points): 좌표를 감싸는 최소한의 타원 계산

line = cv2.fitLine(points, distType, param, reps, aeps, line): 중심점을 통과하는 직선 계산
distType: 거리 계산 방식 (cv2.DIST_L2, cv2.DIST_L1, cv2.DIST_L12, cv2.DIST_FAIR, cv2.DIST_WELSCH, cv2.DIST_HUBER)
param: distType에 전달할 인자, 0 = 최적 값 선택
reps: 반지름 정확도, 선과 원본 좌표의 거리, 0.01 권장
aeps: 각도 정확도, 0.01 권장
line(optional): vx, vy 정규화된 단위 벡터, x0, y0: 중심점 좌표

'''


# In[ ]:


컨투어를 감싸는 도형 그리기 (cntr_bound_fit.py)

import cv2
import numpy as np

# 이미지 읽어서 그레이스케일 변환, 바이너리 스케일 변환
img = cv2.imread("../img/lightning.png")
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, th = cv2.threshold(imgray, 127,255,cv2.THRESH_BINARY_INV)

# 컨튜어 찾기
im, contours, hr = cv2.findContours(th, cv2.RETR_EXTERNAL,                                        cv2.CHAIN_APPROX_SIMPLE)
contr = contours[0]

# 감싸는 사각형 표시(검정색)
x,y,w,h = cv2.boundingRect(contr)
cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,0), 3)

# 최소한의 사각형 표시(초록색)
rect = cv2.minAreaRect(contr)
box = cv2.boxPoints(rect)   # 중심점과 각도를 4개의 꼭지점 좌표로 변환
box = np.int0(box)          # 정수로 변환
cv2.drawContours(img, [box], -1, (0,255,0), 3)

# 최소한의 원 표시(파랑색)
(x,y), radius = cv2.minEnclosingCircle(contr)
cv2.circle(img, (int(x), int(y)), int(radius), (255,0,0), 2)

# 최소한의 삼각형 표시(분홍색)
ret, tri = cv2.minEnclosingTriangle(contr)
cv2.polylines(img, [np.int32(tri)], True, (255,0,255), 2)

# 최소한의 타원 표시(노랑색)
ellipse = cv2.fitEllipse(contr)
cv2.ellipse(img, ellipse, (0,255,255), 3)

# 중심점 통과하는 직선 표시(빨강색)
[vx,vy,x,y] = cv2.fitLine(contr, cv2.DIST_L2,0,0.01,0.01)
cols,rows = img.shape[:2]
cv2.line(img,(0, 0-x*(vy/vx) + y), (cols-1, (cols-x)*(vy/vx) + y),                                                        (0,0,255),2)

# 결과 출력
cv2.imshow('Bound Fit shapes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 컨투어 단순화

# In[ ]:


# 근사 컨투어 (cntr_approximate.py)

import cv2
import numpy as np

img = cv2.imread('../img/bad_rect.png')
img2 = img.copy()

# 그레이스케일과 바이너리 스케일 변환
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
ret, th = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)

# 컨투어 찾기 ---①
temp, contours, hierachy = cv2.findContours(th, cv2.RETR_EXTERNAL,                                      cv2.CHAIN_APPROX_SIMPLE)
contour = contours[0]
# 전체 둘레의 0.05로 오차 범위 지정 ---②
epsilon = 0.05 * cv2.arcLength(contour, True)
# 근사 컨투어 계산 ---③
approx = cv2.approxPolyDP(contour, epsilon, True)

# 각각 컨투어 선 그리기 ---④
cv2.drawContours(img, [contour], -1, (0,255,0), 3)
cv2.drawContours(img2, [approx], -1, (0,255,0), 3)

# 결과 출력
cv2.imshow('contour', img)
cv2.imshow('approx', img2)
cv2.waitKey()
cv2.destroyAllWindows()


# In[ ]:


'''

hull = cv2.convexHull(points, hull, clockwise, returnPoints): 볼록 선체 계산
points: 입력 컨투어
hull(optional): 볼록 선체 결과
clockwise(optional): 방향 지정 (True: 시계 방향)
returnPoints(optional): 결과 좌표 형식 선택 (True: 볼록 선체 좌표 변환, False: 입력 컨투어 중에 볼록 선체에 해당하는 인덱스 반환)

retval = cv2.isContourConvex(contour): 볼록 선체 만족 여부 확인
retval: True인 경우 볼록 선체임

defects = cv2.convexityDefects(contour, convexhull): 볼록 선체 결함 찾기
contour: 입력 컨투어
convexhull: 볼록 선체에 해당하는 컨투어의 인덱스
defects: 볼록 선체 결함이 있는 컨투어의 배열 인덱스, N x 1 x 4 배열, [starts, end, farthest, distance]
  start: 오목한 각이 시작되는 컨투어의 인덱스
  end: 오목한 각이 끝나는 컨투어의 인덱스
  farthest: 볼록 선체에서 가장 먼 오목한 지점의 컨투어 인덱스
  distance: farthest와 볼록 선체와의 거리

'''


# In[ ]:


# 볼록 선체 (cntr_convexhull.py)

import cv2
import numpy as np

img = cv2.imread('../img/hand.jpg')
img2 = img.copy()
# 그레이 스케일 및 바이너리 스케일 변환 ---①
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# 컨투어 찾기와 그리기 ---②
temp, contours, heiarchy = cv2.findContours(th, cv2.RETR_EXTERNAL,                                          cv2.CHAIN_APPROX_SIMPLE)
cntr = contours[0]
cv2.drawContours(img, [cntr], -1, (0, 255,0), 1)

# 볼록 선체 찾기(좌표 기준)와 그리기 ---③
hull = cv2.convexHull(cntr)
cv2.drawContours(img2, [hull], -1, (0,255,0), 1)
# 볼록 선체 만족 여부 확인 ---④
print(cv2.isContourConvex(cntr), cv2.isContourConvex(hull))

# 볼록 선체 찾기(인덱스 기준) ---⑤
hull2 = cv2.convexHull(cntr, returnPoints=False)
# 볼록 선체 결함 찾기 ---⑥
defects = cv2.convexityDefects(cntr, hull2)
# 볼록 선체 결함 순회
for i in range(defects.shape[0]):
    # 시작, 종료, 가장 먼 지점, 거리 ---⑦
    startP, endP, farthestP, distance = defects[i, 0]
    # 가장 먼 지점의 좌표 구하기 ---⑧
    farthest = tuple(cntr[farthestP][0])
    # 거리를 부동 소수점으로 변환 ---⑨
    dist = distance/256.0
    # 거리가 1보다 큰 경우 ---⑩
    if dist > 1 :
        # 빨강색 점 표시 
        cv2.circle(img2, farthest, 3, (0,0,255), -1)
# 결과 이미지 표시
cv2.imshow('contour', img)
cv2.imshow('convex hull', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 컨투어와 도형 매칭

# In[ ]:


# 도형 매칭으로 비슷한 도형 찾기 (contr_matchShape.py)

import cv2
import numpy as np

# 매칭을 위한 이미지 읽기
target = cv2.imread('../img/4star.jpg') # 매칭 대상
shapes = cv2.imread('../img/shapestomatch.jpg') # 여러 도형
# 그레이 스케일 변환
targetGray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
shapesGray = cv2.cvtColor(shapes, cv2.COLOR_BGR2GRAY)
# 바이너리 스케일 변환
ret, targetTh = cv2.threshold(targetGray, 127, 255, cv2.THRESH_BINARY_INV)
ret, shapesTh = cv2.threshold(shapesGray, 127, 255, cv2.THRESH_BINARY_INV)
# 컨투어 찾기
_, cntrs_target, _ = cv2.findContours(targetTh, cv2.RETR_EXTERNAL,                                             cv2.CHAIN_APPROX_SIMPLE)
_, cntrs_shapes, _ = cv2.findContours(shapesTh, cv2.RETR_EXTERNAL,                                             cv2.CHAIN_APPROX_SIMPLE)

# 각 도형과 매칭을 위한 반복문
matchs = [] # 컨투어와 매칭 점수를 보관할 리스트
for contr in cntrs_shapes:
    # 대상 도형과 여러 도형 중 하나와 매칭 실행 ---①
    match = cv2.matchShapes(cntrs_target[0], contr, cv2.CONTOURS_MATCH_I2, 0.0)
    # 해당 도형의 매칭 점수와 컨투어를 쌍으로 저장 ---②
    matchs.append( (match, contr) )
    # 해당 도형의 컨투어 시작지점에 매칭 점수 표시 ---③
    cv2.putText(shapes, '%.2f'%match, tuple(contr[0][0]),                    cv2.FONT_HERSHEY_PLAIN, 1,(0,0,255),1 )
# 매칭 점수로 정렬 ---④
matchs.sort(key=lambda x : x[0])
# 가장 적은 매칭 점수를 얻는 도형의 컨투어에 선 그리기 ---⑤
cv2.drawContours(shapes, [matchs[0][1]], -1, (0,255,0), 3)
cv2.imshow('target', target)
cv2.imshow('Match Shape', shapes)
cv2.waitKey()
cv2.destroyAllWindows()


# In[ ]:


# 출처 : http://localhost:8888/notebooks/kim/OpenCV/OpenCV%20-%2022.%20%EC%BB%A8%ED%88%AC%EC%96%B4(Contour).ipynb


# In[ ]:




