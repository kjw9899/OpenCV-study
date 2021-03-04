#!/usr/bin/env python
# coding: utf-8

# ### 관심 영역(ROI) 표시하기

# In[25]:


# 관심영역 표시

import cv2
import numpy as np

img = cv2.imread('C:/cdd/moons1.jpg')

x = 320; y =150; w = 50; h =50 # roi 좌표
roi = img[x:x+w, y:y+h] # roi 지정

print(roi.shape) # roi shape : (50,50,3)
cv2.rectangle(roi,(0,0), (h-1,w-1), (0,255,0)) # roi 전체에 사각형 그리기
cv2.imshow('img',img)

key = cv2.waitKey(0)
print(key)
cv2.destroyAllWindows()


# In[ ]:


# 관심 영역 복제 및 새 창에 띄우기


# In[36]:


import cv2
import numpy as np

img = cv2.imread('C:/cdd/sunset1.jpg')

x=320; y=150; w=50; h=50
roi = img[y:y+h, x:x+w]     # roi 지정
img2 = roi.copy() # roi 복제

img[y:y+h, x+w:x+w+w] = roi # 새로운 좌표에 roi 추가
cv2.rectangle(img, (x,y), (x+w+w,y+h), (0,255,0)) # 2개의 태양 영역에 사각형 표시

cv2.imshow("img", img)
cv2.imshow("roi", roi)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 마우스 드래그로 관심 영역 표시하기

# In[40]:


# 마우스로 관심영역 지정 및 표시, 저장 (roi_crop_mouse.py)

import cv2
import numpy as np

isDragging = False                      # 마우스 드래그 상태 저장 
x0, y0, w, h = -1,-1,-1,-1              # 영역 선택 좌표 저장
blue, red = (255,0,0),(0,0,255)         # 색상 값 

def onMouse(event,x,y,flags,param):     # 마우스 이벤트 핸들 함수  ---①
    global isDragging, x0, y0, img      # 전역변수 참조
    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 마우스 버튼 다운, 드래그 시작 ---②
        isDragging = True
        x0 = x
        y0 = y
    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스 움직임 ---③
        if isDragging:                  # 드래그 진행 중
            img_draw = img.copy()       # 사각형 그림 표현을 위한 이미지 복제
            cv2.rectangle(img_draw, (x0, y0), (x, y), blue, 2) # 드래그 진행 영역 표시
            cv2.imshow('img', img_draw) # 사각형 표시된 그림 화면 출력
    elif event == cv2.EVENT_LBUTTONUP:  # 왼쪽 마우스 버튼 업 ---④
        if isDragging:                  # 드래그 중지
            isDragging = False          
            w = x - x0                  # 드래그 영역 폭 계산
            h = y - y0                  # 드래그 영역 높이 계산
            print("x:%d, y:%d, w:%d, h:%d" % (x0, y0, w, h))
            if w > 0 and h > 0:         # 폭과 높이가 양수이면 드래그 방향이 옳음 ---⑤
                img_draw = img.copy()   # 선택 영역에 사각형 그림을 표시할 이미지 복제
                # 선택 영역에 빨간 사각형 표시
                cv2.rectangle(img_draw, (x0, y0), (x, y), red, 2) 
                cv2.imshow('img', img_draw) # 빨간 사각형 그려진 이미지 화면 출력
                roi = img[y0:y0+h, x0:x0+w] # 원본 이미지에서 선택 영영만 ROI로 지정 ---⑥
                cv2.imshow('cropped', roi)  # ROI 지정 영역을 새창으로 표시
                cv2.moveWindow('cropped', 0, 0) # 새창을 화면 좌측 상단에 이동
                cv2.imwrite('./cropped.jpg', roi)   # ROI 영역만 파일로 저장 ---⑦
                print("croped.")
            else:
                cv2.imshow('img', img)  # 드래그 방향이 잘못된 경우 사각형 그림ㅇㅣ 없는 원본 이미지 출력
                print("좌측 상단에서 우측 하단으로 영역을 드래그 하세요.")

img = cv2.imread('C:/cdd/sunset1.jpg')
cv2.imshow('img', img)
cv2.setMouseCallback('img', onMouse) # 마우스 이벤트 등록 ---⑧
cv2.waitKey()
cv2.destroyAllWindows()


# ### selectROI 함수

# In[46]:


# selectROI로 관심영역 지정 및 표시, 저장 (roi_select_img.py)

import cv2
import numpy as np

img = cv2.imread('C:/cdd/moons1.jpg')

x,y,w,h = cv2.selectROI('img', img , True)

if w and h :
    roi = img[y:y+h, x:x+w]
    cv2.imshow('crroped', roi) # ROI 지정 영역을 새창으로 표시
    cv2.moveWindow('crroped',0,0) # 새 창을 화면 좌측 상단으로 이동
    cv2.imwrite('C:/cdd/moons1_crroped.jpg',roi) # roi영역만 파일로 저장
    
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


# 출처 : https://bkshin.tistory.com/category/OpenCV

