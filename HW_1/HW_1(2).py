import numpy as np
import cv2 as cv
import sys

# 카메라와 연결 시도 (DirectShow 드라이버 사용)
cap = cv.VideoCapture(0, cv.CAP_DSHOW)

if not cap.isOpened():
    sys.exit('카메라 연결 실패')

while True:
    ret, frame = cap.read()  # 비디오를 구성하는 프레임 획득
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    # 그레이스케일 변환
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Canny 엣지 검출
    canny = cv.Canny(gray, 100, 200)

    # 흑백 이미지를 컬러처럼 만들어서 합치기
    gray_cap = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)

    # 원본 프레임과 에지 검출 결과를 좌우로 붙이기
    caps = np.hstack((frame, gray_cap))

    # 결과 출력
    cv.imshow('Video display', caps)

    # 'q' 키를 누르면 종료
    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
