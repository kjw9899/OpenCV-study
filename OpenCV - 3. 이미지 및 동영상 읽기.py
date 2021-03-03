#!/usr/bin/env python
# coding: utf-8

# # 이미지 읽기

# In[6]:


import cv2

img_profile_photo = "C:/cdd/moons1.jpg" # 이미지를 표시할 경로
img = cv2.imread(img_profile_photo) # 이미지를 읽어서 img 변수에 할당
img = cv2.resize(img,dsize=(1280, 1060), interpolation=cv2.INTER_AREA)

if img is not None:
    cv2.imshow('Profile', img) # 읽은 이미지를 'Profile'이라는 제목으로 화면에 표시
    cv2.waitKey() # 키가 입력될 때까지 대기
    cv2.destroyAllWindows() # 창 모두 닫기
    
else :
    print("No Image")
    
# img=cv2.resize(img,dsize=(1280, 1060), interpolation=cv2.INTER_AREA) : 사진 크기 조정 함수


# ### 이미지 파일을 회색으로 화면에 표시 

# In[15]:


import cv2

img_file = "C:/cdd/smile1.jpg"
img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

if img is not None :
    cv2.imshow('smile', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
else : 
    print('No Image')


# ### 이미지 저장하기

# In[22]:


import cv2

img_file = 'C:/cdd/smile1.jpg'

img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
cv2.imshow('smile', img)
cv2.imwrite(save_file, img)
cv2.waitKey()
cv2.destroyAllWindows()


# ### 동영상 파일 읽기

# In[ ]:


import cv2

video_file = "C:/cdd/face.mp4" # 동영상 파일 경로

cap = cv2.VideoCapture(video_file) # 동영상 캡쳐 객체 생성
if cap.isOpened() : # 캡쳐 초기화 확인
    while True:
        ret, img = cap.read() # 다음 프레임 읽기
        if ret is True :
            cv2.imshow('face',img) # 화면에 표시
            cv2.waitKey() # 25이하면 빠르게, 25 이상이면 느리게
        else :
            cv2.waitKey()
            cv2.destroyAllWindows()
            
else :
    print("can't open video") # 캡쳐 초기화 실패
cap.release() # 캡쳐 자원 반납
cv2.destroyAllWindows()
                
        


# ### 카메라(웹캠) 프레임 읽기

# In[5]:


import cv2

cap = cv2.VideoCapture(0)               # 0번 카메라 장치 연결 -                      # 캡쳐 객체 연결 확인
if cap.isOpened():
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


# ### 웹캠으로 사진찍기

# In[2]:


import cv2 

cap = cv2.VideoCapture(0)
if cap.isOpened() :
    while True :
        ret, frame = cap.read() # 카메라 프레임 읽기
        if ret:
            cv2.imshow('selca', frame) # 프레임 화면에 표시
            if cv2.waitKey(1) != -1:
                cv2.imwrite('C:/cdd/photo.jpg',frame) # 프레임을 'photo.jpg'에 저장
                break
        else :
            print('no frame!')
            break
else:
    print("no camera!")
    
cap.release()
cv2.destroyAllWindows()


# ### 웹캠으로 녹화하기

# In[5]:


import cv2

cap = cv2.VideoCapture(0) # 0번 카메라에 연결
if cap.isOpened:
    file_path = 'C:/cdd/Web_record.mp4' # 저장할 파일 경로 이름
    fps = 33.0
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') #Windowsms는 DIVX
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(width), int(height))
    out = cv2.VideoWriter(file_path, fourcc, fps, size)
    # 비디오 저장을 위한 객체 out을 형성한다.
    while True :
        ret, frame = cap.read()
        if ret is True :
            cv2.imshow('record_me',frame)
            out.write(frame) # 파일 저장, 촬영 되는 영상을 저장하는 객체에 써준다.
            if cv2.waitKey(int(1000/fps)) != -1:
                break
        else :
            print("no frame!")
            break
    out.release()
else :
    print("can't open camera!")
cap.release()
cv2.destroyAllWindows()
    

