import cv2
import mediapipe as mp

# Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Drawing Utils 설정
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 웹캠 열기 (기본 카메라)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("카메라를 불러올 수 없습니다.")
        break

    # BGR → RGB 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 얼굴 랜드마크 검출
    results = face_mesh.process(rgb_frame)

    # 출력 이미지 복사
    annotated_frame = frame.copy()

    # 랜드마크 표시
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 랜드마크를 점으로 시각화
            for idx, landmark in enumerate(face_landmarks.landmark):
                h, w, _ = frame.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(annotated_frame, (x, y), 1, (0, 255, 0), -1)

            # 또는 스타일 있게 그리기 (선택)
            # mp_drawing.draw_landmarks(
            #     image=annotated_frame,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_TESSELATION,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

    # 화면에 출력
    cv2.imshow("Face Mesh (Press ESC to Exit)", annotated_frame)

    # ESC 키 누르면 종료
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
