#!/usr/bin/env python
# coding: utf-8

# ### 직선 그리기

# In[8]:


# 다양한 직선 그리기(draw_line.py)

import cv2

img = cv2.imread('../img/blank_500.jpg')

cv2.line(img, (50, 50), (150, 50), (255,0,0))   # 파란색 1픽셀 선
cv2.line(img, (200, 50), (300, 50), (0,255,0))  # 초록색 1픽셀 선
cv2.line(img, (350, 50), (450, 50), (0,0,255))  # 빨간색 1픽셀 선

# 하늘색(파랑+초록) 10픽셀 선      
cv2.line(img, (100, 100), (400, 100), (255,255,0), 10)          
# 분홍(파랑+빨강) 10픽셀 선      
cv2.line(img, (100, 150), (400, 150), (255,0,255), 10)          
# 노랑(초록+빨강) 10픽셀 선      
cv2.line(img, (100, 200), (400, 200), (0,255,255), 10)          
# 회색(파랑+초록+빨강) 10픽셀 선  
cv2.line(img, (100, 250), (400, 250), (200,200,200), 10)        
# 검정 10픽셀 선    
cv2.line(img, (100, 300), (400, 300), (0,0,0), 10)                    

# 4연결 선
cv2.line(img, (100, 350), (400, 400), (0,0,255), 20, cv2.LINE_4)   
# 8연결 선
cv2.line(img, (100, 400), (400, 450), (0,0,255), 20, cv2.LINE_8)    
# 안티에일리어싱 선 
cv2.line(img, (100, 450), (400, 500), (0,0,255), 20, cv2.LINE_AA)   
# 이미지 전체에 대각선 
cv2.line(img, (0,0), (500,500), (0,0,255))                      

cv2.imshow('lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 사각형 그리기

# In[16]:


# 사각형 그리기(draw_rect.py)

import cv2

img = cv2.imread('../img/blank_500.jpg')

# 좌상, 우하 좌표로 사각형 그리기, 선 두께는 default 1
cv2.rectangle(img, (50, 50), (150, 150), (255,0,0) )        
# 우하, 좌상 좌표로 사각형 그리기, 선 두께 10
cv2.rectangle(img, (300, 300), (100, 100), (0,255,0), 10 )  
# 우상, 좌하 좌표로 사각형 채워 그리기 ---①
cv2.rectangle(img, (450, 200), (200, 450), (0,0,255), -1 )  

cv2.imshow('rectangle', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 다각형 그리기

# In[18]:


# 다각형 그리기(draw_poly)

import cv2
import numpy as np                          # 좌표 표현을 위한 numpy 모듈  ---①

img = cv2.imread('C:/cdd/blank_500.jpg')

# Numpy array로 좌표 생성 ---②
# 번개 모양 선 좌표
pts1 = np.array([[50,50], [150,150], [100,140],[200,240]], dtype=np.int32) 
# 삼각형 좌표
pts2 = np.array([[350,50], [250,200], [450,200]], dtype=np.int32) 
# 삼각형 좌표
pts3 = np.array([[150,300], [50,450], [250,450]], dtype=np.int32) 
# 5각형 좌표
pts4 = np.array([[350,250], [450,350], [400,450], [300,450], [250,350]],                 dtype=np.int32) 

# 다각형 그리기 ---③
cv2.polylines(img, [pts1], False, (255,0,0))       # 번개 모양 선 그리기
cv2.polylines(img, [pts2], False, (0,0,0), 10)     # 3각형 열린 선 그리기 ---④
cv2.polylines(img, [pts3], True, (0,0,255), 10)    # 3각형 닫힌 도형 그리기 ---⑤
cv2.polylines(img, [pts4], True, (0,0,0))          # 5각형 닫힌 도형 그리기

cv2.imshow('polyline', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 원, 타원, 호  그리기

# In[19]:


# 원, 타원, 호 그리기(draw_circle.py)

import cv2

img = cv2.imread('C:/cdd/blank_500.jpg')

# 원점(150,150), 반지름 100 ---①
cv2.circle(img, (150, 150), 100, (255,0,0))     
# 원점(300,150), 반지름 70 ---②
cv2.circle(img, (300, 150), 70, (0,255,0), 5)   
# 원점(400,150), 반지름 50, 채우기 ---③
cv2.circle(img, (400, 150), 50, (0,0,255), -1)  

# 원점(50,300), 반지름(50), 회전 0, 0도 부터 360도 그리기 ---④
cv2.ellipse(img, (50, 300), (50, 50), 0, 0, 360, (0,0,255))    
# 원점(150, 300), 아래 반원 그리기 ---⑤
cv2.ellipse(img, (150, 300), (50, 50), 0, 0, 180, (255,0,0))    
#원점(200, 300), 윗 반원 그리기 ---⑥
cv2.ellipse(img, (200, 300), (50, 50), 0, 181, 360, (0,0,255))    

# 원점(325, 300), 반지름(75,50) 납작한 타원 그리기 ---⑦
cv2.ellipse(img, (325, 300), (75, 50), 0, 0, 360, (0,255,0))    
# 원점(450,300), 반지름(50,75) 홀쭉한 타원 그리기 ---⑧
cv2.ellipse(img, (450, 300), (50, 75), 0, 0, 360, (255,0,255))    

# 원점(50, 425), 반지름(50,75), 회전 15도 ---⑨
cv2.ellipse(img, (50, 425), (50, 75), 15, 0, 360, (0,0,0))    
# 원점(200,425), 반지름(50,75), 회전 45도 ---⑩
cv2.ellipse(img, (200, 425), (50, 75), 45, 0, 360, (0,0,0))    

# 원점(350,425), 홀쭉한 타원 45도 회전 후 아랫 반원 그리기 ---⑪
cv2.ellipse(img, (350, 425), (50, 75), 45, 0, 180, (0,0,255))    
# 원점(400,425), 홀쭉한 타원 45도 회전 후 윗 반원 그리기 ---⑫
cv2.ellipse(img, (400, 425), (50, 75), 45, 181, 360, (255,0,0))    

cv2.imshow('circle', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 글쓰기

# In[23]:


# 글 쓰기 (draw_text.py)

import cv2

img = cv2.imread('C:/cdd/blank_500.jpg')

# sans-serif small
cv2.putText(img, "Plain", (50, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0,0))            
# sans-serif normal
cv2.putText(img, "Simplex", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,0))        
# sans-serif bold
cv2.putText(img, "Duplex", (50, 110), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0,0))         
# sans-serif normall X2  ---①
cv2.putText(img, "Simplex", (200, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,250)) 

# serif small
cv2.putText(img, "Complex Small", (50, 180), cv2.FONT_HERSHEY_COMPLEX_SMALL,             1, (0, 0,0))   
# serif normal
cv2.putText(img, "Complex", (50, 220), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0,0))
# serif bold
cv2.putText(img, "Triplex", (50, 260), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0,0))               
# serif normal X2  ---②
cv2.putText(img, "Complex", (200, 260), cv2.FONT_HERSHEY_TRIPLEX, 2, (0,0,255))               

# hand-wringing sans-serif
cv2.putText(img, "Script Simplex", (50, 330), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,             1, (0, 0,0)) 
# hand-wringing serif
cv2.putText(img, "Script Complex", (50, 370), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,             1, (0, 0,0)) 

# sans-serif + italic ---③
cv2.putText(img, "Plain Italic", (50, 430),             cv2.FONT_HERSHEY_PLAIN | cv2.FONT_ITALIC, 1, (0, 0,0)) 
# sarif + italic
cv2.putText(img, "Complex Italic", (50, 470),             cv2.FONT_HERSHEY_COMPLEX | cv2.FONT_ITALIC, 1, (0, 0,0)) 

cv2.imshow('draw text', img)
cv2.waitKey()
cv2.destroyAllWindows()


# In[ ]:




