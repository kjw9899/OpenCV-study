# 애완 동물을 소개해 주세요
# animal = "여자친구"
# name = "우만이"
# age = 26
# hobby = "쇼핑"
# is_adult = age >= 3

# '''이렇게 하면
# 여러문장이
# 주석처리가 됩니다.
# '''

# print("우리집 " + animal + "의 이름은 " + name + "이에요")
# hobby = "낮잠"
# print(name,"는 " , age , "이며, " , hobby , "을 아주 좋아해요")
# print(name + "는 어른일까요? " + str(is_adult))

'''

# Quiz) 변수를 이용하여 다음 문장을 출력하시오

# 변수명 : station

# 변수값
# : "사당" "신도림" "인천공항"

# 출력문장
# : XX행 열차가 들어오고 있습니다.

# '''

# station ="신도림"

# print(station+"행 열차가 들어오고 있습니다.")

# print(1+1)
# print(3-1)
# print(5*2)
# print(6/3)

# print(2**3)
# print(5%3)
# print(10%3)
# print(5//3)
# print(10//3)

# print(3==2)
# print(4+2==6)

# print(1 != 3)
# print(not (1 != 3))

# print((3>0) and (3<5))
# print((3 > 0 ) & (3<5))

# print((3 > 0) or (3>5))
# print((3 > 0) | (3 >5 ))

# number = 2 + 3 * 4
# print(number)
# number = number + 2
# print(number)
# number +=2 # 18
# print(number)

# number *=2
# print(number)
# number /=2
# print(number)
# number -= 2
# print(number)
# number%=5
# print(number)
'''
# 숫자 처리 함수

print(abs(-5))
print(pow(4,2))
print(max(5,10))
print(min(5,2))
print(round(3.14))
print(round(4.123))

from math import *
print(floor(4.99)) #내림
print(ceil(3.14)) #올림
print(sqrt(16))
'''

# 랜덤 함수


# print(random()) # 0.0이상 1.0이하의 임의의 값 생성
# print(random() * 10)
# print(int(random() * 10))
# print(int(random() * 10))
# print(int(random() * 10))

# # print(int(random() * 10) +)

# print(int(random() * 45) + 1)
# print(int(random() * 45) + 1)
# print(int(random() * 45) + 1)
# print(int(random() * 45) + 1)
# print(int(random() * 45) + 1)
# print(int(random() * 45) + 1)

# print(random())


#  from random import *

# print(randrange(1,46)) # 1- 45 미만의 값 생성

# from random import *

# # print(random())

# print(randrange(1,46))
# print(randrange(1,46))
# print(randrange(1,46))
# print(randrange(1,46))
# print(randrange(1,46))
# print(randrange(1,46))

# print(randint(1,45)) # 1부터 45이하의 값을 생성한다.

'''

Quiz) 당신은 최근에 코딩 스터디 모임을 새로 만들었습니다
월 4회 스터디를 하는데 3번은 온라인으로 하고 1번은 오프라인으로 하기로 했습니다.
아래 조건에 맞는 오프라인 모임 날짜를 정해주는 프로그램을 작성하시오.

조건1 : 랜덤으로 날짜를 뽑아야함
조건2 : 월별 날짜는 다름을 감안하여 최소 일수인 28일 이내로 정해야함
조건3: 매월 1~3일은  스터디를 준비해야하므로 제외

(출력문 예제)
오프라인 스터디 모임 날짜는 매월 x일로 선정되었습니다.



# from random import *
# number = randint(4,28)
# print("오프라인 스터디 모임 날짜는 매월" + str(number + "일로 선정되었습니다.")
'''
# 문자열

# sentence = 'Im a boy'
# print(sentence)
# sentence2 = "파이썬은 쉬워요"
# print(sentence2)
# sentence3 = '''
# 나는 소년이고,
# 파이썬은 쉬워요
# '''
# print(sentence3)

# 슬라이싱

# jumin = "980704-1111416"

# print("성별 : " + jumin[7])
# print("연 : " + jumin[0:2] ) # 0부터 2직전까지 (0,1)
# print("월 : " + jumin[2:4])
# print("일 : " + jumin[4:6])
# print("지역번호 : " + jumin[9:15])
# print("생년월일 : " + jumin[:6]) #처음부터 6직전까지 값을 가져온다
# print("뒤 7자리 :" + jumin[7:])
# print("뒤 7자리(뒤에부터) : " + jumin[-7:])
# # 맨 뒤에서 7번째 끝까지

# 문자열 처리 함수

# python = "Python is Amazing"

# print(python.lower())
# print(python.upper())
# print(python[0].islower())
# print(python[0].isupper())
# print(len(python))
# print(python.replace("Python","Java"))

# index = python.index("n")
# print(index)
# index = python.index("n", index + 1)
# print(index)

# print(python.find("Java"))
# #print(python.index("Java"))

# print(python.count("n"))

# 문자열 포멧

# # 방법 1
# print("나는 %d살 입니다." % 24)
# print("나는 %s를 좋아해요" %"파이썬")
# print("Apple 이라는 단어는 %c로 시작해요" % "A" )

# print("나는 %s색과 %s색을 좋아해요" % ("파란", "빨강"))

# # 방법 2

# print("나는 {}살입니다." .format(24))
# print("나는 {}색과 {}색을 좋아해요." .format("빨강", "파란"))
# print("나는 {0}색과 {1}색을 좋아해요." .format("빨강", "파란"))
# print("나는 {1}색과 {0}색을 좋아해요." .format("빨강", "파란"))

# # 방법 3

# print("나는 {age}살이며, {color}색을 좋아해요." .format(age = 20, color = "빨간"))
# print("나는 {age}살이며, {color}색을 좋아해요." .format(color = "빨간", age = 24))

# # 방법 4
# age = 24
# color = "빨간"
# print(f"나는 {age}살이며, {color}색을 좋아해요.")

# 문자 탈출 \n

# print("백문이 불여일견 \n벽견이 불여일타")

# # 저는 김재원입니다.
# print(나는 "김재원" 입니다')

# print("나는 \"김재원\"입니다.")

# # \\ : 문장 내에서 하나의 \로 바뀜

# print("C:\\Users\\kjw98\\OneDrive\\바탕 화면\\pythonWorkspace>")

# \r : 커서를 맨앞으로 이동
#print("Red Apple\rPine")

# \b : 백스페이스 (한글자 삭제)
#print("Red d\bApple")

# \t : 탭
#print("Red\t Apple")

'''
Quiz) 사이트 별로 비밀번호를 만들어 주는 프로그램을 작성하시오

예) http://naver.com
규칙1 : hppt:// 부분은 제외
규칙2 : 처음 만나는 점(.) 이후 부분 제외 => naver
규칙3 : 남은 글자 중 처음 세자리 + 글자갯수 + 글자 내 'e' 갯수 + "!" 로 구성
           (nav)                   (5)          (1)              (1)
예 생성된 비밀번호 : nav51!

'''

# url = "http://google.com"
# my_str = url.replace("http://", "")
# # print(my_str)

# my_str = my_str[:my_str.index(".")]
# # print(my_str)

# # pw_first3 = my_str[:3]
# # print(pw_first3)

# # pw_len = len(my_str)
# # print(pw_len)

# # print("생성된 비밀번호 : %s%s%s%s" % (pw_first3, pw_len, my_str.count("e"),"!"))

# password = my_str[:3] + str(len(my_str)) + str(my_str.count("e")) + "!"


# print("{}의 비밀번호는 {}입니다." .format (url, password))

# 리스트 : 순서를 가지는 객체의 집합

# 지하철 칸별로 10명 20명 30명

subway1 = 10
subway2 = 20
subway3 = 30

subway = [10, 20, 30]
print(subway)

subway = ["유재석", "조세호", "박명수"]
print(subway)

# 조세호씨가 몇 번째 칸에 타고 있는가?

print(subway.index("조세호"))

# 하하씨가 다음 정류장에서 다음 칸에 탐

subway.append("하하")
print(subway)
