#!/usr/bin/env python
# coding: utf-8

# Reading Image   

# In[1]:


import cv2

img_file = "C:/cdd/223.jpg" # 표시할 이미지 경로            ---①
img = cv2.imread(img_file)    # 이미지를 읽어서 img 변수에 할당 ---②

if img is not None:
  cv2.imshow('IMG', img)      # 읽은 이미지를 화면에 표시      --- ③
  cv2.waitKey()               # 키가 입력될 때 까지 대기      --- ④
  cv2.destroyAllWindows()     # 창 모두 닫기            --- ⑤
else:
    print('No image file.')


# Reading Gray Image

# In[7]:


import cv2

img_file = "C:/cdd/223.jpg"
img = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE)

if img is not None:
  cv2.imshow('IMG', img)      # 읽은 이미지를 화면에 표시      --- ③
  cv2.waitKey()               # 키가 입력될 때 까지 대기      --- ④
  cv2.destroyAllWindows()     # 창 모두 닫기            --- ⑤
else:
    print('No image file.')


# Save Image

# In[9]:


import cv2

img_file = 'C:/cdd/223.jpg'
save_file = 'C:/cdd/223.jpg'

img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
cv2.imshow(img_file, img)
cv2.imwrite(save_file, img) #파일로 저장, 포맷은 확장에 따름
cv2.waitKey()
cv2.destroyAllWindows()


# Reading video

# In[ ]:


# 동영상 파일 읽기 (video_play.py)

import cv2

video_file = "C:/cdd/ysl.mp4" # 동영상 파일 경로

cap = cv2.VideoCapture(video_file) # 동영상 캡쳐 객체 생성  ---①
if cap.isOpened():                 # 캡쳐 객체 초기화 확인
    while True:
        ret, img = cap.read()      # 다음 프레임 읽기      --- ②
        if ret:                     # 프레임 읽기 정상 # 프레임을 잘 읽었다면 ret은 True, img는 프레임 이미지가 됩니다.
            cv2.imshow(video_file, img) # 화면에 표시  --- ③
            cv2.waitKey(25)            # 25ms 지연(40fps로 가정)   --- ④
        else:                       # 다음 프레임 읽을 수 없슴,
            break                   # 재생 완료
else:
    print("can't open video.")      # 캡쳐 객체 초기화 실패
cap.release()                       # 캡쳐 자원 반납
cv2.destroyAllWindows()


# Reading camera frame

# In[34]:


import cv2

cap = cv2.VideoCapture(0)               # 0번 카메라 장치 연결 ---①
if cap.isOpened():                      # 캡쳐 객체 연결 확인
    while True:
        ret, img = cap.read()           # 다음 프레임 읽기
        if ret:
            cv2.imshow('camera', img)   # 다음 프레임 이미지 표시
            if cv2.waitKey(1) != -1 :    # 1ms 동안 키 입력 대기 ---②
                break                   # 아무 키라도 입력이 있으면 중지
        else:
            print('no frame')
            break
else:
    print("can't open camera.")
cap.release()                           # 자원 반납
cv2.destroyAllWindows()


# Take picture with Webcam

# In[ ]:


# 웹캠으로 사진찍기 (video_cam_take_pic.py)

import cv2

cap = cv2.VideoCapture(0)                       # 0번 카메라 연결
if cap.isOpened() :
    while True:
        ret, frame = cap.read()                 # 카메라 프레임 읽기
        if ret:
            cv2.imshow('camera',frame)          # 프레임 화면에 표시
            if cv2.waitKey(1) != -1:            # 아무 키나 누르면
                cv2.imwrite('photo.jpg', frame) # 프레임을 'photo.jpg'에 저장
                break
        else:
            print('no frame!')
            break
else:
    print('no camera!')
cap.release()
cv2.destroyAllWindows()


# 웹캠으로 녹화하기

# In[46]:


# 웹캠으로 녹화하기 (video_cam_rec.py)

import cv2

cap = cv2.VideoCapture(0)    # 0번 카메라 연결
if cap.isOpened:
    file_path = 'C:/cdd/face.mp4'    # 저장할 파일 경로 이름 ---①
    fps = 30.0                     # FPS, 초당 프레임 수
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') # 인코딩 포맷 문자
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) #cap.get()은 동영상이나 카메라의 속성을 확인하는 함수
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(width), int(height))                        # 프레임 크기
    out = cv2.VideoWriter(file_path, fourcc, fps, size) # VideoWriter 객체 생성
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('camera-recording',frame)
            out.write(frame)                        # 파일 저장
            if cv2.waitKey(int(1000/fps)) != -1: 
                break
        else:
            print("no frame!")
            break
    out.release()                                   # 파일 닫기
else:
    print("can't open camera!")
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




