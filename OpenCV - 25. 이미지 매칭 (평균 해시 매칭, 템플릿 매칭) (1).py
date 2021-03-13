#!/usr/bin/env python
# coding: utf-8

# ### 이미지 매칭

# In[ ]:


'''

이미지 매칭이란?
서로 다른 두 이미지를 비교해서 짝이 맞는 같은 형대의 객체가 있는지 찾아내는 기술
쉽게 말해 두 이미지 간 유사도를 측정하는 작업입니다.
특징을 대표할 수 있는 숫자를 특징 혹은 특징 벡터 혹은 특징 디스크렙터라고 합니다.

'''


# ### 평균 해시 매칭

# In[ ]:


'''
평균 해시 매칭은 이미지 매칭의 한 기법인데, 효과는 떨어지지만 구현이 아주 간단합니다. 
평균 해시 매칭은 특징 벡터를 구하기 위해 평균갑을 사용합니다.
우선, 두 이미지 사이에서 비슷한 그림을 찾기 전에 찾고자 하는 그림의 특징 벡터를 구하는 방법을 알아보겠습니다.

1. 이미지를 가로 세로 비율과 무관하게 특정한 크기로 축소합니다.
2. 픽셀 전체의 평균값을 구해서 각 픽셀의 값이 평균보다 작으면 0, 크면 1로 바꿉니다.
3. 0 또는 1로만 구성된 각 픽셀 값을 1행 1열로 변환합니다. (이는 한 개의 2진수 숫자로 볼 수 있습니다.)

이때 비교하고자 하는 두 이미지를 같은 크기로 축소해야합니다. 그렇기 때문에 0과 1의 개수도 동일합니다.
(2진수로 표현했을 때 비트 개수가 같다고 볼 수 있습니다.) 2진수가 너무 길어서 보기 불편하다면 필요에 따라 
10진수나 16진수 등으로 변환해서 사용할 수 있습니다.

'''


# In[9]:


import cv2

#영상 읽어서 그레이 스케일로 변환
img = cv2.imread('C:/cdd/pistol.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 8*8 크기로 축소
gray = cv2.resize(gray, (16,16))

# 영상의 평균 값 구하기
avg = gray.mean()

# 평균값을 기준으로 0과 1로 변횐
bin = 1 * (gray > avg)
print(bin)


# In[10]:


'''

이렇게 해서 얻은 평균 해시를 다른 이미지 것과 비교해서 얼마나  비슷한지를 알아내야 합니다. 
비슷한 정도를 측정하는 방법에는 여러가지 방법이 있습니다.
그중 가장 대표적인 것이 유클리드 거리와 해밍거리입니다.

유클리드 거리는 두 값의 차이로 거리를 계산합니다.
예를 들어 5와 비교할 값으로 1과 7이 있다면 5와 1의 유클리드 거리는 5-1=4 이고
5와 7의 유클리드 거리는 7-5=2입니다.
유클리드 거리가 작을수록 두 수는 비슷한 수라고 판단하므로 5는 1보다는 7과 더 유사하다고 결론 짓습니다.

해밍거리는 두 값의 길이가 같아야 계산할 수 있습니다.
해밍거리는 두 수의 같은 자이 값 중 서로 다른 것이 몇 개인지를 판단하여 유사도를 계산합니다.
예를 들어 12345와 비교할 값으로 12354와 92345가 잇을 때 1234와 12354의 마지막 자이가 45와 54로
다르므로 해밍리는 2입니다. 반면 1234와 9234는 1과 9 한자리만 다르므로 해밍거리는 1입니다.
따라서 1234는 12343보다 92345와 더 유사하다고 판단합니다.


앞서 구한 권총의 평균 해시를 다른 이미지와비교할 때는 해밍거리를 서야합니다.
유클리드 거리는 자릿수가 높을수록 차이가 크게 벌어지지만 해밍거리는 몇 개의 숫자가 다른가만 고려하기 때문입니다/.
이미지를 비교하는데 평균 해시 숫자의 크기가 중요하기보다는 얼마나 유사한 자릿수가 많은지가 더 중요합니다.

이제 권총의 이미지의 평균 해시를 다른 이미지의 평균해시와 해밍거리로 비교해 유사도를 측정해보겟습니다.
우선 여러 이미지가 필요한데 아래 링크를 총해 다운로드할 수 있습니다.

'''


# In[ ]:


# 시뮬 이미지 중에서 권총이미지 찾기

import cv2
import numpy as np
import glob

# 영상 읽기 및 표시
img = cv2.imread('C:/cdd/pistol.jpg')
cv2.imshow('query'.img)

# 비교할 영상들이 있는 경로
search_dir = '../img/101_ObjectCategories'

# 이미지를 16*16크기의 평균 해시로 변환
def img2hash(img) :
    gray = cv2.cvtColot(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (16,16))
    avg = gray.mean()
    bi = 1 * (gray > avg)
    return bi

# 해밍거리 측정 함수
def hamming_distance(a,b):
    a = a.reshape(1,-1)
    b = b.reshape(1,-1)
    # 같은 자리의 값이 서로 다른 것들의 합
    distance = (a != b).sum()
    return distance

# 권총 영상의 해쉬 구하기
query_hash = img2hash(img)

# 이미지 데이터 셋 디렉토리의 모든 영상 파일 경로
img_path = glob.glob(search_dir+'/**/*.jpg')
for path in img_path:
    # 데이타 셋 영상 한개 읽어서 표시 ---⑥
    img = cv2.imread(path)
    cv2.imshow('searching...', img)
    cv2.waitKey(5)
    # 데이타 셋 영상 한개의 해시  ---⑦
    a_hash = img2hash(img)
    # 해밍 거리 산출 ---⑧
    dst = hamming_distance(query_hash, a_hash)
    if dst/256 < 0.25: # 해밍거리 25% 이내만 출력 ---⑨
        print(path, dst/256)
        cv2.imshow(path, img)
cv2.destroyWindow('searching...')
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### 템플릿 매칭

# In[ ]:


'''

템플릿 매칭은 특정 물체에 대한 이미지를 준비해 두고 그 물테가 포함되어 있을 것이라고 예상할 수 있는
이미지와 비교하여 매칭 되는 위치를 찾는 것입니다. 이때 미리 준비한 이미지를 템플릿 이미지라고 합니다. 
템플릿 이미지는 비교할 이미지보다 크기가 항상 작아야 합니다. 템플릿 매칭과 관련한 함수는 다음과 같습니다.

result = cv2.matchTemplate(img, templ, method, result, mask)
img: 입력 이미지
templ: 템플릿 이미지
method: 매칭 메서드 (cv2.TM_SQDIFF(가장 어두운 곳이 매칭 지정, 가장 박은 곳이 매칭 지정): 제곱 차이 매칭, 완벽 매칭:0, 나쁜 매칭: 큰 값 /
cv2.TM_SQDIFF_NORMED(cv2.TM_SQDIFF(가장 어두운 곳이 매칭 지정, 가장 박은 곳이 매칭 지정): 제곱 차이 매칭의 정규화 / 
cv2.TM_CCORR: 상관관계 매칭, 완벽 매칭: 큰 값, 나쁜 매칭: 0 / 
cv2.TM_CCORR_NORMED: 상관관계 매칭의 정규화 / 
cv2.TM_CCOEFF: 상관계수 매칭, 완벽 매칭:1, 나쁜 매칭: -1 / 
cv2.TM_CCOEFF_NORMED: 상관계수 매칭의 정규화)
result(optional): 매칭 결과, (W - w + 1) x (H - h + 1) 크기의 2차원 배열
[여기서 W, H는 입력 이미지의 너비와 높이, w, h는 템플릿 이미지의 너비와 높이]
mask(optional): TM_SQDIFF, TM_CCORR_NORMED인 경우 사용할 마스크


cv2.matchTemplate() 함수는 입력이미지(img)에서 템플릿 이미지를 슬라이딩하면서 주어진 메서드에 따라 
매칭을 수행합니다.cv2.matchTemplate() 함수의 반환 값은 (W - w + 1) * (H - h +1) 크기의 2차원 배열입니다.
(W,H는 입력이미지의 너비와 높이, w,h는 템플릿 이미지의 너비와 높이)
이 배열의 최대, 최소 값을 구하면 원하는 최선의 매칭 값과 매칭점을 구할 수 있습니다.
이것을 손쉽게 해주는 함수가 바로 cv2.minMaxLoc()입니다. 이 함수는 입력 배열에서의 최소, 최대 뿐만 아니라
최소 값, 최대 값의 좌표도 반환합니다. 

'''


# In[ ]:


# 템플릿 매칭으로 객체 위치 검출

import cv2
import numpy as np

# 입력이미지와 템플릿 이미지 읽기
img = cv2.imread('C:/cdd/taekwonv1.jpg')
template = cv2.imread('../img/taekwonv1.jpg')
th, tw = template.shape[:2]
cv2.imshow('temlate', template)

# 3가지 매칭 메서드 순환
methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED',                                      'cv2.TM_SQDIFF_NORMED']']
for i, method_name in enumerate(methods):
    img_draw = img.copy()
    method = eval(method_name)
    # 템플릿 매칭
   res = cv2.matchTemplate(img, template, method)
    # 최대, 최소값과 그 좌표 구하기 ---②
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(method_name, min_val, max_val, min_loc, max_loc)

    # TM_SQDIFF의 경우 최소값이 좋은 매칭, 나머지는 그 반대 ---③
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
        match_val = min_val
    else:
        top_left = max_loc
        match_val = max_val
    # 매칭 좌표 구해서 사각형 표시   ---④      
    bottom_right = (top_left[0] + tw, top_left[1] + th)
    cv2.rectangle(img_draw, top_left, bottom_right, (0,0,255),2)
    # 매칭 포인트 표시 ---⑤
    cv2.putText(img_draw, str(match_val), top_left,                 cv2.FONT_HERSHEY_PLAIN, 2,(0,255,0), 1, cv2.LINE_AA)
    cv2.imshow(method_name, img_draw)
cv2.waitKey(0)
cv2.destroyAllWindows()    


# In[ ]:


# 출처 : https://bkshin.tistory.com/entry/OpenCV-25-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EB%A7%A4%EC%B9%AD-%ED%8F%89%EA%B7%A0-%ED%95%B4%EC%8B%9C-%EB%A7%A4%EC%B9%AD-%ED%85%9C%ED%94%8C%EB%A6%BF-%EB%A7%A4%EC%B9%AD?category=1148027

