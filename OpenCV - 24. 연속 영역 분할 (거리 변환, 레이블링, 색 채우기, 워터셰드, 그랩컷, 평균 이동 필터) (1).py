#!/usr/bin/env python
# coding: utf-8

# ### 거리 변환

# In[ ]:


'''

cv2.distanceTransform(src, distanceType, maskSize)
src: 입력 영상, 바이너리 스케일
distanceType: 거리 계산 방식 (cv2.DIST_L2, cv2.DIST_L1, cv2.DIST_L12, cv2.DIST_FAIR, cv2.DIST_WELSCH, cv2.DIST_HUBER)
maskSize: 거리 변환 커널 크기

Adaptive Method는 아래와 같습니다.
cv2.ADAPTIVE_THRESH_MEAN_C : 주변영역의 평균값으로 결정
cv2.ADAPTIVE_THRESH_GAUSSIAN_C : blockSize 영역의 모든 픽셀에 중심점으로부터의 거리에 대한 가우시안 가중치 적용

'''


# In[1]:


# 거리 변환으로 읽어서 바이너리 스케일로 변환

import cv2
import numpy as np

# 이미지를 읽어서 바이너리 스케일로 변환
img = cv2.imread('C:/cdd/full_body.jpg', cv2.IMREAD_GRAYSCALE)
_, biimg = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# 거리 변환
dst = cv2.distanceTransform(biimg, cv2.DIST_L2, 5)
# 거리 값을 0 ~ 255 범위로 정규화
dst = (dst/(dst.max()-dst.min()) * 255).astype(np.uint8)
# 거리 값에 쓰레시홀드로 완전한 뼈대 찾기
skeleton = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, -3)

# 결과 출력
cv2.imshow('origin', img)
cv2.imshow('dist', dst)
cv2.imshow('skel', skeleton)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 레이블링

# In[ ]:


'''

연결된 요소끼리 분리하는 방법 중 레이블링이라는 방법이 있습니다. 
아래와 같이 이미지에서 픽셀 값이 0으로 끊어지지 않는 부분끼리 같은 값을 부여해서 분리할 수 있습니다.
OpenCV에서 제공하는 cv2.connectedComponents() 함수를 활용하면 이를 구현할 수 있습니다.
이 함수는 이미지 전체에서 0으로 끊어지지 않는 부분끼리 같은 값을 부여합니다.


retval, labels = cv2.connectedComponents(src, labels, connectivity=8, ltype): 연결 요소 레이블링과 개수 반환

src: 입력 이미지, 바이너리 스케일
labels(optional): 레이블링된 입력 이미지와 같은 크기의 배열
connectivity(optional): 연결성을 검사할 방향 개수(4, 8 중 선택)
ltype(optional): 결과 레이블 배열 dtype
retval(optional): 레이블 개수

retval, labels, stats, centroids = cv2.connectedComponentsWithStats(src, labels, stats, centroids, connectivity, ltype): 레이블링된 각종 상태 정보 반환
stats: N x 5 행렬 (N: 레이블 개수) [x좌표, y좌표, 폭, 높이, 너비]
centroids: 각 레이블의 중심점 좌표, N x 2 행렬 (N: 레이블 개수)
cv2.connectedComponents() 함수를 활용해서 연결된 요소끼리 같은 색상을 칠해보겠습니다. 
주석 처리된 cv2.connectedComponentsWithStats()로 코드를 돌려도 동일한 결과가 나올 겁니다.


'''


# In[6]:


# 연결된 영역 레이블링

import cv2
import numpy as np

# 이미지 읽기
img = cv2.imread('C:/cdd/donut.jpg')
# 결과 이미지 생성
img2 = np.zeros_like(img)
# 그레이 스케일과 바이너리 스케일로 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 연결된 요소 레이블링 적용
cnt, labels = cv2.connectedComponents(th)

# 레이블 갯수 만큼 순회
for i in range(cnt) :
    # 레이블이 같은 영역에 랜덤한 색상 적용
    img2[labels == i] = [int(j) for j in np.random.randint(0,255, 3)]
    
# 결과 출력
# 결과 출력
cv2.imshow('origin', img)
cv2.imshow('labeled', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 색채우기

# In[7]:


'''

retval, img, mask, rect = cv2.floodFill(img, mask, seed, newVal, loDiff, upDiff, flags)
img: 입력 이미지, 1 또는 3채널
mask: 입력 이미지보다 2 x 2 픽셀이 더 큰 배열, 0이 아닌 영역을 만나면 채우기 중지
seed: 채우기 시작할 좌표
newVal: 채우기에 사용할 색상 값
loDiff, upDiff(optional): 채우기 진행을 결정할 최소/최대 차이 값
flags(optional): 채우기 방식 선택 (cv2.FLOODFILL_MASK_ONLY: img가 아닌 mask에만 채우기 적용, cv2.FLOODFILL_FIXED_RANGE: 이웃 픽셀이 아닌 seed 픽셀과 비교)
retval: 채우기 한 픽셀의 개수
rect: 채우기가 이루어진 영역을 감싸는 사각형

이 함수는 img 이미지의 seed(시작) 좌표부터 시작해서 newval의 값으로 채우기 시작합니다.
이때 이웃하는 픽셀에 채우기를 계속하려면 현재 픽셀이 이웃 픽셀의 loDiff를 뺀 값보다 크거나 같고
upDiff를 더한 값 보다 작거나 같아야합니다.
이것을 식으로 정리하면 위와 같습니다.
(만약 loDiff와 upDiff를 생략하면 seed의 픽셀값과 같은 값을 갖는 이웃 픽셀만 채우기를 진행합니다.)

loDiff, upDiff = 채우기 진행을 결정할 최소/최대 차이 값



'''


# In[1]:


# 마우스로 색 채우기

import cv2
import numpy as np

img = cv2.imread('C:/cdd/taekwonv1.jpg')
rows, cols = img.shape[:2]
# 마스크 생성, 원래 이미지 보다 2 픽셀 크게
mask = np.zeros((rows+2, cols+2), np.uint8)
# 채우기에 사용할 색
newVal = (255,255,255)
# 최소, 최대 차이 값
loDiff, upDiff = (10,10,10), (10,10,10)

# 마우스 이벤트 처리 함수
def onMouse(event, x, y, flags, param) :
    global mask, img
    if event == cv2.EVENT_LBUTTONDOWN :
        seed = (x,y)
        # 색 채우기 적용
        retval = cv2.floodFill(img, mask, seed, newVal, loDiff, upDiff)
        # 채우기  변경 결과 표시
        cv2.imshow('img', img)
        
# 화면 출력
cv2.imshow('img', img)
cv2.setMouseCallback('img', onMouse)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 워터셰드

# In[ ]:


'''

markers = cv2.watershed(img, markers)
img: 입력 이미지
markers: 마커, 입력 이미지와 크기가 같은 1차원 배열(int32)

markers는 입력 이미지와 행과 열 크기가 같은 1차원 배열로 전달해야 합니다.
markers의 값은 경계를 찾고자 하는 픽셀 영역은 -1을 갖게 하고 나머지 연결된 영역에 대해서는 
동일한 정수 값을 갖게 합니다.
예를 들어 1은 배경, 2는 전경인 식입니다. cv2.watershed() 함수를 활용해 경계를 나눠보겠습니다. 

'''


# In[9]:


# 마우스와 워터셰드로 배경 분리

import cv2
import numpy as np

img = cv2.imread('C:/cdd/1.jpg')
rows, cols = img.shape[:2]
img_draw = img.copy()

# 마커 생성, 모든 요소는 0으로 초기화
marker = np.zeros((rows, cols), np.int32)
markerId = 1 # 마커아이디는 1에서 시작
colors = [] # 마커 선택한 영역 색상 저장할 공간
isDragging = False # 드래그 여부 확인 변수

# 마우스 이벤트 처리 함수
def onMouse(event, x, y, flags, param):
    global img_draw, marker, markerId, isDragging
    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 마우스 버튼 다운, 드래그 시작 
        isDragging = True
        # 각 마커의 아이디와 현 위치의 색상 값을 쌍으로 매핑해서 저장 
        colors.append((markerId, img[y,x]))
    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스 움직임
        if isDragging:                  # 드래그 진행 중
            # 마우스 좌표에 해당하는 마커의 좌표에 동일한 마커 아이디로 채워 넣기 ---②
            marker[y,x] = markerId
            # 마커 표시한 곳을 빨강색점으로 표시해서 출력
            cv2.circle(img_draw, (x,y), 3, (0,0,255), -1)
            cv2.imshow('watershed', img_draw)
    elif event == cv2.EVENT_LBUTTONUP:  # 왼쪽 마우스 버튼 업
        if isDragging:                  
            isDragging = False          # 드래그 중지
            # 다음 마커 선택을 위해 마커 아이디 증가 ---③
            markerId +=1
    elif event == cv2.EVENT_RBUTTONDOWN: # 오른쪽 마우스 버튼 누름
            # 모아 놓은 마커를 이용해서 워터 쉐드 적용 ---④
            cv2.watershed(img, marker)
            # 마커에 -1로 표시된 경계를 초록색으로 표시  ---⑤
            img_draw[marker == -1] = (0,255,0)
            for mid, color in colors: # 선택한 마커 아이디 갯수 만큼 반복
                # 같은 마커 아이디 값을 갖는 영역을 마커 선택한 색상으로 채우기 ---⑥
                img_draw[marker==mid] = color
            cv2.imshow('watershed', img_draw) # 표시한 결과 출력

# 화면 출력
cv2.imshow('watershed', img)
cv2.setMouseCallback('watershed', onMouse)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 그랩컷

# In[ ]:


'''

그랩컷은 사용자가 전경(배경이 아닌 부분)으로 분리할 부분에 사각형 표시를 해주면
전경과 배경의 색상 분토를 추정해서 동일한 레이블을 가진 영역에서 전경과 배경을 분리합니다.

mask, bgdModel, fgdModel = cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode)
img: 입력 이미지
mask: 입력 이미지와 크기가 같은 1 채널 배열, 배경과 전경을 구분하는 값을 저장 (cv2.GC_BGD: 확실한 배경(0), cv2.GC_FGD: 확실한 전경(1), cv2.GC_PR_BGD: 아마도 배경(2), cv2.GC_PR_FGD: 아마도 전경(3))
rect: 전경이 있을 것으로 추측되는 영역의 사각형 좌표, 튜플 (x1, y1, x2, y2)
bgdModel, fgdModel: 함수 내에서 사용할 임시 배열 버퍼 (재사용할 경우 수정하지 말 것)
iterCount: 반복 횟수
mode(optional): 동작 방법 (cv2.GC_INIT_WITH_RECT: rect에 지정한 좌표를 기준으로 그랩컷 수행, cv2.GC_INIT_WITH_MASK: mask에 지정한 값을 기준으로 그랩컷 수행, cv2.GC_EVAL: 재시도)





'''


# In[ ]:


# 마우스와 그랩컷으로 배경과 전경 분리 (grabcut.py)

import cv2
import numpy as np

img = cv2.imread('C:/cdd/1.jpg')
img_draw = img.copy()
mask = np.zeros(img.shape[:2], dtype=np.uint8)  # 마스크 생성
rect = [0,0,0,0]    # 사각형 영역 좌표 초기화
mode = cv2.GC_EVAL  # 그랩컷 초기 모드
# 배경 및 전경 모델 버퍼
bgdmodel = np.zeros((1,65),np.float64)
fgdmodel = np.zeros((1,65),np.float64)

# 마우스 이벤트 처리 함수
def onMouse(event, x, y, flags, param):
    global mouse_mode, rect, mask, mode
    if event == cv2.EVENT_LBUTTONDOWN : # 왼쪽 마우스 누름
        if flags <= 1: # 아무 키도 안 눌렀으면
            mode = cv2.GC_INIT_WITH_RECT # 드래그 시작, 사각형 모드 ---①
            rect[:2] = x, y # 시작 좌표 저장
    # 마우스가 움직이고 왼쪽 버튼이 눌러진 상태
    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON :
        if mode == cv2.GC_INIT_WITH_RECT: # 드래그 진행 중 ---②
            img_temp = img.copy()
            # 드래그 사각형 화면에 표시
            cv2.rectangle(img_temp, (rect[0], rect[1]), (x, y), (0,255,0), 2)
            cv2.imshow('img', img_temp)
        elif flags > 1: # 키가 눌러진 상태
            mode = cv2.GC_INIT_WITH_MASK    # 마스크 모드 ---③
            if flags & cv2.EVENT_FLAG_CTRLKEY :# 컨트롤 키, 분명한 전경
                # 흰색 점 화면에 표시
                cv2.circle(img_draw,(x,y),3, (255,255,255),-1)
                # 마스크에 GC_FGD로 채우기      ---④
                cv2.circle(mask,(x,y),3, cv2.GC_FGD,-1)
            if flags & cv2.EVENT_FLAG_SHIFTKEY : # 쉬프트키, 분명한 배경
                # 검정색 점 화면에 표시
                cv2.circle(img_draw,(x,y),3, (0,0,0),-1)
                # 마스크에 GC_BGD로 채우기      ---⑤
                cv2.circle(mask,(x,y),3, cv2.GC_BGD,-1)
            cv2.imshow('img', img_draw) # 그려진 모습 화면에 출력
    elif event == cv2.EVENT_LBUTTONUP: # 마우스 왼쪽 버튼 뗀 상태 ---⑥
        if mode == cv2.GC_INIT_WITH_RECT : # 사각형 그리기 종료
            rect[2:] =x, y # 사각형 마지막 좌표 수집
            # 사각형 그려서 화면에 출력 ---⑦
            cv2.rectangle(img_draw, (rect[0], rect[1]), (x, y), (255,0,0), 2)
            cv2.imshow('img', img_draw)
        # 그랩컷 적용 ---⑧
        cv2.grabCut(img, mask, tuple(rect), bgdmodel, fgdmodel, 1, mode)
        img2 = img.copy()
        # 마스크에 확실한 배경, 아마도 배경으로 표시된 영역을 0으로 채우기
        img2[(mask==cv2.GC_BGD) | (mask==cv2.GC_PR_BGD)] = 0
        cv2.imshow('grabcut', img2) # 최종 결과 출력
        mode = cv2.GC_EVAL # 그랩컷 모드 리셋
# 초기 화면 출력 및 마우스 이벤트 등록
cv2.imshow('img', img)
cv2.setMouseCallback('img', onMouse)
while True:    
    if cv2.waitKey(0) & 0xFF == 27 : # esc
        break
cv2.destroyAllWindows()


# In[ ]:


# 출처 : http://localhost:8888/notebooks/kim/OpenCV/OpenCV%20-%2024.%20%EC%97%B0%EC%86%8D%20%EC%98%81%EC%97%AD%20%EB%B6%84%ED%95%A0%20(%EA%B1%B0%EB%A6%AC%20%EB%B3%80%ED%99%98%2C%20%EB%A0%88%EC%9D%B4%EB%B8%94%EB%A7%81%2C%20%EC%83%89%20%EC%B1%84%EC%9A%B0%EA%B8%B0%2C%20%EC%9B%8C%ED%84%B0%EC%85%B0%EB%93%9C%2C%20%EA%B7%B8%EB%9E%A9%EC%BB%B7%2C%20%ED%8F%89%EA%B7%A0%20%EC%9D%B4%EB%8F%99%20%ED%95%84%ED%84%B0).ipynb#

