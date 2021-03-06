#!/usr/bin/env python
# coding: utf-8

# ### 2차원 히스토그램

# In[4]:


# 2D 히스토그램

import cv2
import matplotlib.pylab as plt

plt.style.use('classic')
img = cv2.imread('C:/cdd/mask1.jpg')

plt.subplot(131)
hist = cv2.calcHist([img], [0,1], None, [32,32], [0,256,0,256])
p = plt.imshow(hist)
plt.title('Blue and Green')                                     #--④
plt.colorbar(p)

plt.subplot(132)
hist = cv2.calcHist([img], [1,2], None, [32,32], [0,256,0,256]) #--⑥
p = plt.imshow(hist)
plt.title('Green and Red')
plt.colorbar(p)

plt.subplot(133)
hist = cv2.calcHist([img], [0,2], None, [32,32], [0,256,0,256]) #--⑦
p = plt.imshow(hist)
plt.title('Blue and Red')
plt.colorbar(p)

plt.show()


# ### 역투영

# In[5]:


# 마우스로 선택한 영역의 물체 분리하기 (histo_backproject.py)

import cv2
import numpy as np
import matplotlib.pyplot as plt

win_name = 'back_projection'
img = cv2.imread('C:/cdd/smile2.jpg')
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
draw = img.copy()

#--⑤ 역투영된 결과를 마스킹해서 결과를 출력하는 공통함수
def masking(bp, win_name):
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cv2.filter2D(bp,-1,disc,bp)
    _, mask = cv2.threshold(bp, 1, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow(win_name, result)

#--⑥ 직접 구현한 역투영 함수
def backProject_manual(hist_roi):
    #--⑦ 전체 영상에 대한 H,S 히스토그램 계산
    hist_img = cv2.calcHist([hsv_img], [0,1], None,[180,256], [0,180,0,256])
    #--⑧ 선택영역과 전체 영상에 대한 히스토그램 그램 비율계산
    hist_rate = hist_roi/ (hist_img + 1)
    #--⑨ 비율에 맞는 픽셀 값 매핑
    h,s,v = cv2.split(hsv_img)
    bp = hist_rate[h.ravel(), s.ravel()]
    # 비율은 1을 넘어서는 안되기 때문에 1을 넘는 수는 1을 갖게 함
    bp = np.minimum(bp, 1)
    # 1차원 배열을 원래의 shape로 변환
    bp = bp.reshape(hsv_img.shape[:2])
    cv2.normalize(bp,bp, 0, 255, cv2.NORM_MINMAX)
    bp = bp.astype(np.uint8)
    #--⑩ 역 투영 결과로 마스킹해서 결과 출력
    masking(bp,'result_manual')
 
# OpenCV API로 구현한 함수 ---⑪ 
def backProject_cv(hist_roi):
    # 역투영 함수 호출 ---⑫
    bp = cv2.calcBackProject([hsv_img], [0, 1], hist_roi,  [0, 180, 0, 256], 1)
    # 역 투영 결과로 마스킹해서 결과 출력 ---⑬ 
    masking(bp,'result_cv')

# ROI 선택 ---①
(x,y,w,h) = cv2.selectROI(win_name, img, False)
if w > 0 and h > 0:
    roi = draw[y:y+h, x:x+w]
    # 빨간 사각형으로 ROI 영역 표시
    cv2.rectangle(draw, (x, y), (x+w, y+h), (0,0,255), 2)
    #--② 선택한 ROI를 HSV 컬러 스페이스로 변경
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    #--③ H,S 채널에 대한 히스토그램 계산
    hist_roi = cv2.calcHist([hsv_roi],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    #--④ ROI의 히스토그램을 매뉴얼 구현함수와 OpenCV 이용하는 함수에 각각 전달
    backProject_manual(hist_roi)
    backProject_cv(hist_roi)
cv2.imshow(win_name, draw)
cv2.waitKey()
cv2.destroyAllWindows()


# In[4]:


'''
cv2.getStructuringElement()를 활용해 구조요소를 생성합니다.

cv2.getStructuringElement(커널의 형태, 커널의 크기, 중심점)로 구조 요소을 생성합니다.

커널의 형태는 직사각형(Rect), 십자가(Cross), 타원(Ellipse)이 있습니다.

커널의 크기는 구조 요소의 크기를 의미합니다. 이때, 커널의 크기가 너무 작다면 커널의 형태는 영향을 받지 않습니다.

고정점은 커널의 중심 위치를 나타냅니다. 필수 매개변수가 아니며, 설정하지 않을 경우 사용되는 함수에서 값이 결정됩니다.

-------------------------------------------------------------------------------------------------------------------------------------

cv2.calcBackProjection(images, channels, hist, ranges, scale, dst=None) -> dst

• images: 입력 영상 리스트
• channels: 역투영 계산에 사용할 채널 번호 리스트
• hist: 입력 히스토그램 (numpy.ndarray)
• ranges: 히스토그램 각 차원의 최솟값과 최댓값으로 구성된 리스트
• scale: 출력 역투영 행렬에 추가적으로 곱할 값
• dst: 출력 역투영 영상. 입력 영상과 동일 크기, cv2.CV_8U.




''''''


# In[ ]:

# 출처 : https://bkshin.tistory.com/entry/OpenCV-11-2%EC%B0%A8%EC%9B%90-%ED%9E%88%EC%8A%A4%ED%86%A0%EA%B7%B8%EB%9E%A8%EA%B3%BC-%EC%97%AD%ED%88%AC%EC%98%81back-project?category=1148027


