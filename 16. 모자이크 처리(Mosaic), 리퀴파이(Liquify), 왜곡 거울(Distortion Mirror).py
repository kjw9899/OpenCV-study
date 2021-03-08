#!/usr/bin/env python
# coding: utf-8

# ### 실습 1 : 모자이크 처리

# In[2]:


# 모자이크 처리

import cv2 

rate = 15 # 모자이크에 사용할 축소 비율 (1/rate)
win_title = 'mosaic'
img = cv2.imread('C:/cdd/man_face.jpg')

while True :
    x,y,w,h = cv2.selectROI(win_title, img, False) # 관심영역 선택
    if w and h :
        roi = img[y:y+h, x:x+w] # 관심영역 지정
        roi = cv2.resize(roi, (w//rate, h//rate)) # 1/rate 비율로 축소
        #원래 크기로 확대
        roi = cv2.resize(roi, (w,h), interpolation = cv2. INTER_AREA)
        img[y:y+h, x:x+w] = roi #원본 이미지에 적용
        cv2.imshow(win_title, img)
    else :
        break

cv2.destroyAllWindows()


# ### 실습2 : 리퀴파이 도구

# In[ ]:


# 포토샵 리퀴파이 도구 (workshop_liquify_tool.py)

import cv2
import numpy as np

win_title = 'Liquify'   # 창 이름
half = 50               # 관심 영역 절반 크기
isDragging = False      # 드래그 여부 플래그

# 리퀴파이 함수
def liquify(img, cx1,cy1, cx2,cy2) :
    # 대상 영역 좌표와 크기 설정
    x, y, w, h = cx1-half, cy1-half, half*2, half*2
    # 관심 영역 설정
    roi = img[y:y+h, x:x+w].copy()
    out = roi.copy()

    # 관심영역 기준으로 좌표 재 설정
    offset_cx1,offset_cy1 = cx1-x, cy1-y
    offset_cx2,offset_cy2 = cx2-x, cy2-y
    
    # 변환 이전 4개의 삼각형 좌표
    tri1 = [[ (0,0), (w, 0), (offset_cx1, offset_cy1)], # 상,top
            [ [0,0], [0, h], [offset_cx1, offset_cy1]], # 좌,left
            [ [w, 0], [offset_cx1, offset_cy1], [w, h]], # 우, right
            [ [0, h], [offset_cx1, offset_cy1], [w, h]]] # 하, bottom

    # 변환 이후 4개의 삼각형 좌표
    tri2 = [[ [0,0], [w,0], [offset_cx2, offset_cy2]], # 상, top
            [ [0,0], [0, h], [offset_cx2, offset_cy2]], # 좌, left
            [ [w,0], [offset_cx2, offset_cy2], [w, h]], # 우, right
            [ [0,h], [offset_cx2, offset_cy2], [w, h]]] # 하, bottom

    
    for i in range(4):
        # 각각의 삼각형 좌표에 대해 어핀 변환 적용
        matrix = cv2.getAffineTransform( np.float32(tri1[i]),                                          np.float32(tri2[i]))
        warped = cv2.warpAffine( roi.copy(), matrix, (w, h),             None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        # 삼각형 모양의 마스크 생성
        mask = np.zeros((h, w), dtype = np.uint8)
        cv2.fillConvexPoly(mask, np.int32(tri2[i]), (255,255,255))
        
        # 마스킹 후 합성
        warped = cv2.bitwise_and(warped, warped, mask=mask)
        out = cv2.bitwise_and(out, out, mask=cv2.bitwise_not(mask))
        out = out + warped

    # 관심 영역을 원본 영상에 합성
    img[y:y+h, x:x+w] = out
    return img 

# 마우스 이벤트 핸들 함수
def onMouse(event,x,y,flags,param):     
    global cx1, cy1, isDragging, img      # 전역변수 참조
    # 마우스 중심 점을 기준으로 대상 영역 따라다니기
    if event == cv2.EVENT_MOUSEMOVE:  
        if not isDragging :
            img_draw = img.copy()       
            # 드래그 영역 표시
            cv2.rectangle(img_draw, (x-half, y-half),                     (x+half, y+half), (0,255,0)) 
            cv2.imshow(win_title, img_draw) # 사각형 표시된 그림 화면 출력
    elif event == cv2.EVENT_LBUTTONDOWN :   
        isDragging = True                   # 드래그 시작
        cx1, cy1 = x, y                     # 드래그 시작된 원래의 위치 좌표 저장
    elif event == cv2.EVENT_LBUTTONUP :
        if isDragging:
            isDragging = False              # 드래그 끝
            # 드래그 시작 좌표와 끝난 좌표로 리퀴파이 적용 함수 호출
            liquify(img, cx1, cy1, x, y)    
            cv2.imshow(win_title, img)

if __name__ == '__main__' :
    img = cv2.imread("C:/cdd/232.jpg")
    h, w = img.shape[:2]

    cv2.namedWindow(win_title)
    cv2.setMouseCallback(win_title, onMouse) 
    cv2.imshow(win_title, img)
    while True:
        key = cv2.waitKey(1)
        if key & 0xFF == 27:
            break
    cv2.destroyAllWindows()


# In[ ]:


# 왜곡 거울 카메라 (workshop_distortion_camera.py)

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
WIDTH = 500
HEIGHT = 300
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
rows, cols = HEIGHT, WIDTH
map_y, map_x = np.indices((rows, cols), dtype=np.float32)

# 거울 왜곡 효과 
map_mirrorh_x,map_mirrorh_y = map_x.copy(), map_y.copy() 
map_mirrorv_x,map_mirrorv_y = map_x.copy(), map_y.copy()    
## 좌우 대칭 거울 좌표 연산
map_mirrorh_x[: , cols//2:] = cols - map_mirrorh_x[:, cols//2:]-1
## 상하 대칭 거울 좌표 연산
map_mirrorv_y[rows//2:, :] = rows - map_mirrorv_y[rows//2:, :]-1
# 물결 효과
map_wave_x, map_wave_y = map_x.copy(), map_y.copy()
map_wave_x = map_wave_x + 15*np.sin(map_y/20)
map_wave_y = map_wave_y + 15*np.sin(map_x/20)    


# 렌즈 효과
## 렌즈 효과, 중심점 이동
map_lenz_x = 2*map_x/(cols-1)-1
map_lenz_y = 2*map_y/(rows-1)-1
## 렌즈 효과, 극좌표 변환
r, theta = cv2.cartToPolar(map_lenz_x, map_lenz_y)
r_convex = r.copy()
r_concave = r.copy()
## 볼록 렌즈 효과 매핑 좌표 연산
r_convex[r< 1] = r_convex[r<1] **2  
print(r.shape, r_convex[r<1].shape)
## 오목 렌즈 효과 매핑 좌표 연산
r_concave[r< 1] = r_concave[r<1] **0.5
## 렌즈 효과, 직교 좌표 복원
map_convex_x, map_convex_y = cv2.polarToCart(r_convex, theta)
map_concave_x, map_concave_y = cv2.polarToCart(r_concave, theta)
## 렌즈 효과, 좌상단 좌표 복원
map_convex_x = ((map_convex_x + 1)*cols-1)/2
map_convex_y = ((map_convex_y + 1)*rows-1)/2
map_concave_x = ((map_concave_x + 1)*cols-1)/2
map_concave_y = ((map_concave_y + 1)*rows-1)/2

while True:
    ret, frame = cap.read()
    frame = frame[:HEIGHT, :WIDTH]
    # 준비한 매핑 좌표로 영상 효과 적용
    mirrorh=cv2.remap(frame,map_mirrorh_x,map_mirrorh_y,cv2.INTER_LINEAR)
    mirrorv=cv2.remap(frame,map_mirrorv_x,map_mirrorv_y,cv2.INTER_LINEAR)
    wave = cv2.remap(frame,map_wave_x,map_wave_y,cv2.INTER_LINEAR,                     None, cv2.BORDER_REPLICATE)
    convex = cv2.remap(frame,map_convex_x,map_convex_y,cv2.INTER_LINEAR)
    concave = cv2.remap(frame,map_concave_x,map_concave_y,cv2.INTER_LINEAR)
    # 영상 합치기
    r1 = np.hstack(( frame, mirrorh, mirrorv))
    r2 = np.hstack(( wave, convex, concave))
    merged = np.vstack((r1, r2))

    cv2.imshow('distorted', merged)
    if cv2.waitKey(1) & 0xFF== 27:
        break
cap.release
cv2.destroyAllWindows()


# In[ ]:




