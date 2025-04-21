# CV_HW_8

## 📂 과제 1 –  SORT 알고리즘을 활용한 다중 객체 추적기 구현

### 📌 주요 코드
```python
# 클래스 이름 불러오기
with open("model/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# YOLOv4 모델 불러오기
net = cv2.dnn.readNet("model/yolov4.weights", "model/yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# SORT 추적기 초기화
tracker = Sort()

# 비디오 파일 열기
video_path = "HW_8/slow_traffic_small.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    height, width = frame.shape[:2]

    # YOLO 입력 전처리
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    # 탐지 결과 저장
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 비최대 억제(NMS) 적용
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    dets = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            dets.append([x, y, x + w, y + h, confidences[i]])
    dets = np.array(dets)

    # SORT 추적기 업데이트
    tracked_objects = tracker.update(dets)

    # 추적된 객체 시각화
    for *bbox, obj_id in tracked_objects:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {int(obj_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 결과 프레임 출력
    cv2.imshow("YOLOv4 + SORT Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```
```python
실행 방법
model/ 폴더에 아래 3개의 파일을 넣는다.
yolov4.cfg, yolov4.weights, coco.names 

필수 라이브러리 설치: pip install opencv-python numpy
실행: python tracker_yolov4_sort.py
```

### ✅ 구현 결과
![스크린샷 2025-04-16 112814](https://github.com/user-attachments/assets/7123a0fa-785f-4679-aa36-5ade16a67e5a)

![스크린샷 2025-04-16 112848](https://github.com/user-attachments/assets/9226a9df-5cf0-4016-91f8-4d5f7653f53a)



## 📂 과제 2 – Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화

### 📌 주요 코드
```python
import cv2
import mediapipe as mp
#OpenCV로 웹캠 영상 처리. Google의 MediaPipe 라이브러리로 얼굴의 468개 랜드마크 검출.

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,          # 실시간 처리 (False)
    max_num_faces=1,                  # 한 화면에서 탐지할 얼굴 수
    refine_landmarks=True,           # 눈, 입술, 홍채 정밀 추출
    min_detection_confidence=0.5     # 최소 탐지 신뢰도
)
# 실시간 영상에서 얼굴을 최대 1명까지 인식하도록 설정. 눈/입/홍채의 세밀한 포인트까지 추출하는 refine_landmarks 기능 사용.

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# MediaPipe에서 제공하는 스타일 기반 시각화 유틸을 불러옴.

cap = cv2.VideoCapture(0)  # 기본 카메라 열기

while cap.isOpened():
    ret, frame = cap.read()
    ...
# 웹캠으로부터 프레임을 실시간으로 받아옴.

rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb_frame)
# OpenCV는 기본적으로 BGR 채널이므로, FaceMesh 입력을 위해 RGB로 변환. FaceMesh 모델에 현재 프레임을 넣어 얼굴 랜드마크 추출 수행.

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for idx, landmark in enumerate(face_landmarks.landmark):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(annotated_frame, (x, y), 1, (0, 255, 0), -1)
# 검출된 얼굴마다 468개의 랜드마크를 (x, y) 좌표로 변환하여 초록 점으로 표시. landmark.x와 landmark.y는 0~1 사이의 상대 좌표이므로, 이미지 크기를 곱해 픽셀 좌표로 변환함.

cv2.imshow("Face Mesh (Press ESC to Exit)", annotated_frame)
if cv2.waitKey(1) == 27:
    break
#ESC 키(27)를 누르면 프로그램 종료. 실시간으로 얼굴 위에 랜드마크 점이 그려진 영상 표시.
```

### ✅ 구현 결과
+ <img width="475" alt="image" src="https://github.com/user-attachments/assets/788bff6b-f2b6-4cef-9a97-170c2c68801a" />
