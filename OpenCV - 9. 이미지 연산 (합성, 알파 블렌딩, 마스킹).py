#!/usr/bin/env python
# coding: utf-8

# ### 이미지 연산

# In[ ]:


'''

cv2.add(src1,src2,dest,mask,dtype) : src1과 src2 더하기
(첫 번째 입력 이미지, 두 번째 입력이미지, 출력 영상, mask 값이 0이 아닌 픽셀만 연산
, 출력 데이터 타입)
cv2.subtract(src1, src2, dest, mask, dtype): src1에서 src2 빼기
모든 파라미터는 cv2.add()와 동일
cv2.multiply(src1, src2, dest, scale, dtype): src1과 src2 곱하기
scale(optional): 연산 결과에 추가 연산할 값
cv2.divide(src1, src2, dest, scale, dtype): src1을 src2로 나누기
모든 파라미터는 cv2.multiply()와 동일
'''


# In[3]:


# 이미지의 사칙 연산 (arithmatic.py)

import cv2
import numpy as np

# ---① 연산에 사용할 배열 생성
a = np.uint8([[200, 50]]) 
b = np.uint8([[100, 100]])

#---② NumPy 배열 직접 연산
add1 = a + b
sub1 = a - b
mult1 = a * 2
div1 = a / 3

# ---③ OpenCV API를 이용한 연산
add2 = cv2.add(a, b)
sub2 = cv2.subtract(a, b)
mult2 = cv2.multiply(a , 2)
div2 = cv2.divide(a, 3)

#---④ 각 연산 결과 출력
print(add1, add2)
print(sub1, sub2)
print(mult1, mult2)
print(div1, div2)


# In[4]:


# mask와 누적 할당 연산 (arithmatic_mask.py)

import cv2
import numpy as np

#---① 연산에 사용할 배열 생성
a = np.array([[1, 2]], dtype=np.uint8)
b = np.array([[10, 20]], dtype=np.uint8)

# 2번째 요소가 0인 마스크 배열 생성
mask = np.array([[1,0]], dtype = np.uint8)

# 누적 할당과의 비교 연산
c1 = cv2.add(a,b,None,mask)
print(c1)
c2 = cv2.add(a,b,b.copy(),mask)
print(c2,b)
c3 = cv2.add(a,b,b,mask)
print(c3,b)


# ### 이미지 합성

# In[10]:


import cv2
import numpy as np
import matplotlib.pylab as plt

img1 = cv2.imread('C:/cdd/2321.jpg')
img2 = cv2.imread('C:/cdd/smile2.jpg')

img3 = img1 + img2
img4 = cv2.add(img1,img2)

imgs = {'img1': img1, 'img2':img2, 'img1 + img2:':img3, 'cv.add(img1,img2):':img4}

for i, (k,v) in enumerate(imgs.items()) :
    plt.subplot(2,2,i+1)
    plt.imshow(v[:,:,::-1])
    plt.title(k)
    plt.xticks([]); plt.yticks([])

plt.show()


# ### 가중치 추가

# In[1]:


'''

cv2.addWeight(img1, alpha, img2, beta, gamma)
img1, img2: 합성할 두 이미지
alpha: img1에 지정할 가중치(알파 값)
beta: img2에 지정할 가중치, 흔히 (1-alpha) 적용
gamma: 연산 결과에 가감할 상수, 흔히 0 적용

'''


# In[9]:


import cv2
import numpy as np

alpha = 0.1 #사용할 알파 값

# 합성에 사용할 영상 읽기

img1 = cv2.imread('C:/cdd/2321.jpg')
img2 = cv2.imread('C:/cdd/smile2.jpg')

# Numpy 배열에 수식을 직접 연산해서 알파 블렌딩 적용
blended = img1 * alpha + img2 * (1-alpha)
blended = blended.astype(np.uint8) # 소수점 발생을 제거하기 위함
cv2.imshow('img1 * alpha + img2 * (1-alpha)', blended)

# cv2.addWeighted() 함수 알파 블렌딩 적용
dst = cv2.addWeighted(img1, alpha, img2, (1-alpha), 0)
cv2.imshow('cv2.addWeighted', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 트랙바로 알파 블렌딩

# In[13]:


import cv2
import numpy as np

win_name = 'Alpha blending'     # 창 이름
trackbar_name = 'fade'          # 트렉바 이름

# ---① 트렉바 이벤트 핸들러 함수
def onChange(x):
    alpha = x/100
    dst = cv2.addWeighted(img1, 1-alpha, img2, alpha, 0) 
    cv2.imshow(win_name, dst)


img1 = cv2.imread('C:/cdd/2321.jpg')
img2 = cv2.imread('C:/cdd/smile2.jpg')

cv2.imshow(win_name, img1)
cv2.createTrackbar(trackbar_name,win_name,0,100,onChange)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 비트와이즈 연산

# In[16]:


import numpy as np , cv2
import matplotlib.pylab as plt

# 연산에 사용할 이미지 생성
img1 = np.zeros( (200,400), dtype=np.uint8)
img2 = np.zeros( (200,400), dtype=np.uint8)
img1[:, :200] = 255 # 왼쪽은 흰색, 오른쪽은 검정색
img2[100:200, :] = 255

#비트와이저 연산
bitAnd = cv2.bitwise_and(img1, img2)
bitOr = cv2.bitwise_or(img1, img2)
bitXor = cv2.bitwise_xor(img1, img2)
bitNot = cv2.bitwise_not(img1)

imgs = {'img1':img1, 'img2':img2, 'and':bitAnd, 
          'or':bitOr, 'xor':bitXor, 'not(img1)':bitNot}

for i, (title, img) in enumerate(imgs.items()):
    plt.subplot(3,2,i+1)
    plt.title(title)
    plt.imshow(img,'gray')
    plt.xticks([]); plt.yticks([])
    
plt.show()


# ### bitwise_and 연산으로 마스킹하기 (bitwise_masking.py)

# In[18]:


import numpy as np
import cv2
import matplotlib.pylab as plt

img = cv2.imread('C:/cdd/moons1.jpg')

# 마스크 만들기
mask = np.zeros_like(img)
cv2.circle(mask, (260,210), 100, (255,255,255), -1)

masked = cv2.bitwise_and(img, mask)

cv2.imshow('original', img)
cv2.imshow('mask', mask)
cv2.imshow('masked',masked)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 두 이미지의 차이

# In[19]:


# 두 이미지의 차를 통해 도면의 차이 찾아내기 (diff_absolute.py)

import numpy as np, cv2

#--① 연산에 필요한 영상을 읽고 그레이스케일로 변환
img1 = cv2.imread('C:/cdd/moons1.jpg')
img2 = cv2.imread('C:/cdd/smile2.jpg')
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#--② 두 영상의 절대값 차 연산
diff = cv2.absdiff(img1_gray, img2_gray)

#--③ 차 영상을 극대화 하기 위해 쓰레시홀드 처리 및 컬러로 변환
_, diff = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY)
diff_red = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
diff_red[:,:,2] = 0

#--④ 두 번째 이미지에 변화 부분 표시
spot = cv2.bitwise_xor(img2, diff_red)

#--⑤ 결과 영상 출력
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('diff', diff)
cv2.imshow('spot', spot)
cv2.waitKey()
cv2.destroyAllWindows()


# ### 이미지 합성과 마스킹

# In[21]:


import cv2
import numpy as np

# 압성에 사용할 영상 읽기, 전경 영상은 4채널  png 파일
img_fg = cv2.imread('C:/cdd/OpenCV_LOGO1.png', cv2.IMREAD_UNCHANGED)
img_bg = cv2.imread('C:/cdd/2321.jpg')

# 알파채널을 이용해서 마스크와 역마스크 생성
_, mask = cv2.threshold(img_fg[:,:,3],1,255,cv2.THRESHOLD_BINARY)#배경 전경 분리
mask_inv = cv2.bitwise_not(mask)

# 전경 영상 크기로 배경 영상에서 ROI 잘라내기
img_fg = cv2.cvtColor(img_fg, cv2.COLOR_BGR2BGR)
h, w = img_fg.shape[:2]
roi = img_bg[10:10+h, 10:10+w]

# 마스크를 이용해서 오려내기
masked_fg = cv2.bitwise_and(img_fg, img_fg, mask=mask)
masked_bg = cv2.bitwise_and()

# 이미지 합성
added = masked_fg + masked_bg
img_bg[10:10+h, 10:10+w] = added

cv2.imshow('mask', mask)
cv2.imshow('mask_inv', mask_inv)
cv2.imshow('masked_fg', masked_fg)
cv2.imshow('masked_bg', masked_bg)
cv2.imshow('added', added)
cv2.imshow('result', img_bg)
cv2.waitKey()
cv2.destroyAllWindows()


# In[ ]:


'''

_, mask = cv2.threshod(img_fg[:, :, 3], 1, 255, cv2.THRESH_BINARY)를 호출하여 
배경과 전경을 분리하는 마스크를 만듭니다.
OpenCV라고 쓰여있는 이미지는 배경이 투명합니다.
따라서 배경 부분은 BRGA의 A값이 0입니다. 
반면 배경이 아닌 전경 부분은 A가 0이 아닙니다.
따라서 A가 1 이상이면 255, 1 미만이면 0으로 바꾸어주면 배경은 검은색, 
전경은 흰색이 됩니다. 
mask_inv = cv2.bitwise_not(mask)이므로 mask_inv는 mask의 반대입니다.
즉, 배경은 흰색, 전경은 검은색입니다.
이 두 mask를 활용하여 여수 이미지와 OpenCV 이미지를 합성했습니다.

'''


# In[ ]:


# HSV 색상으로 마스킹 (hsv_color_mask.py)

import cv2
import numpy as np
import matplotlib.pylab as plt

#--① 큐브 영상 읽어서 HSV로 변환
img = cv2.imread("../img/cube.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#--② 색상별 영역 지정
blue1 = np.array([90, 50, 50])
blue2 = np.array([120, 255,255])
green1 = np.array([45, 50,50])
green2 = np.array([75, 255,255])
red1 = np.array([0, 50,50])
red2 = np.array([15, 255,255])
red3 = np.array([165, 50,50])
red4 = np.array([180, 255,255])
yellow1 = np.array([20, 50,50])
yellow2 = np.array([35, 255,255])

# --③ 색상에 따른 마스크 생성
mask_blue = cv2.inRange(hsv, blue1, blue2)
mask_green = cv2.inRange(hsv, green1, green2)
mask_red = cv2.inRange(hsv, red1, red2)
mask_red2 = cv2.inRange(hsv, red3, red4)
mask_yellow = cv2.inRange(hsv, yellow1, yellow2)

#--④ 색상별 마스크로 색상만 추출
res_blue = cv2.bitwise_and(img, img, mask=mask_blue)
res_green = cv2.bitwise_and(img, img, mask=mask_green)
res_red1 = cv2.bitwise_and(img, img, mask=mask_red)
res_red2 = cv2.bitwise_and(img, img, mask=mask_red2)
res_red = cv2.bitwise_or(res_red1, res_red2)
res_yellow = cv2.bitwise_and(img, img, mask=mask_yellow)

#--⑤ 결과 출력
imgs = {'original': img, 'blue':res_blue, 'green':res_green, 
                            'red':res_red, 'yellow':res_yellow}
for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(2,3, i+1)
    plt.title(k)
    plt.imshow(v[:,:,::-1])
    plt.xticks([]); plt.yticks([])
plt.show()


# In[ ]:


# 크로마 키 마스킹과 합성 (chromakey.py)

import cv2
import numpy as np
import matplotlib.pylab as plt

#--① 크로마키 배경 영상과 합성할 배경 영상 읽기
img1 = cv2.imread('../img/man_chromakey.jpg')
img2 = cv2.imread('../img/street.jpg')

# ROI 선택을 위한 좌표 계산
height1, width1 = img1.shape[:2]
height1, width2 = img2.shape[:2]
x = (width2 -width1)//2
y = height2 - height1
w = x + width1
h = y + height1

#--③ 크로마키 배경 영상에서 크로마키 영역을 10픽셀 정도로 지정
chromakey = img1[:10, :10, :]
offset = 20

#--④ 크로마키 영역과 영상 전체를 HSV로 변경
hsv_chroma = cv2.cvtColor(chromakey, cv2.COLOR_BGR2HSV)
hsv_img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

#--⑤ 크로마키 영역의 H값에서 offset 만큼 여유를 두어서 범위 지정
# offset 값은 여러차례 시도 후 결정
#chroma_h = hsv_chroma[0]
chroma_h = hsv_chroma[:,:,0]
lower = np.array([chroma_h.min()-offset, 100, 100])
upper = np.array([chroma_h.max()+offset, 255, 255])

#--⑥ 마스크 생성 및 마스킹 후 합성
mask = cv2.inRange(hsv_img, lower, upper)
mask_inv = cv2.bitwise_not(mask)
roi = img2[y:h, x:w]
fg = cv2.bitwise_and(img1, img1, mask=mask_inv)
bg = cv2.bitwise_and(roi, roi, mask=mask)
img2[y:h, x:w] = fg + bg

#--⑦ 결과 출력
cv2.imshow('chromakey', img1)
cv2.imshow('added', img2)
cv2.waitKey()
cv2.destroyAllWindows()

# 출처 : https://bkshin.tistory.com/category/OpenCV
