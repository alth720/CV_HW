# CV_HW_5

## 📂 과제 1 –  간단한 이미지 분류기 구현

### 📌 주요 코드
```python
# MNIST 데이터셋 불러오기
# 데이터를 train/test 세트로 자동 분리해줌
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 전처리
# 픽셀값을 0~1 범위로 정규화
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
# 28×28 이미지를 1차원 벡터(784차원)로 변환
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)
# 정답 라벨을 one-hot 인코딩으로 변환
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 신경망 모델 구성
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),  # 은닉층
    layers.Dense(64, activation='relu'),                       # 은닉층
    layers.Dense(10, activation='softmax')                     # 출력층
]) # 입력층: 784차원 입력 / 은닉층: ReLU 활성화 함수 사용 / 출력층: softmax 함수로 10개 클래스 확률 출력

# 모델 컴파일 (옵티마이저, 손실함수, 평가 지표 설정) 
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# 최적화: Adam / 손실 함수: 다중 클래스 분류용 cross entropy / 평가 지표: 정확도

# 모델 훈련 (History 객체 받아오기)
# 총 5에폭 동안 학습 / 배치 사이즈 32 / 훈련 데이터의 10%를 검증용으로 사용
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1) 

# 테스트 세트로 정확도 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"테스트 정확도: {test_acc:.4f}")

# 학습 정확도와 검증 정확도를 에폭 단위로 그래프로 시각화
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
```

### ✅ 구현 결과
<img width="599" alt="image" src="https://github.com/user-attachments/assets/38b2c5a3-e01c-453e-828e-d27856788e82" />
<img width="116" alt="image" src="https://github.com/user-attachments/assets/efa883ca-4943-4fe8-84fb-849f09fe8679" />


## 📂 과제 2 –  CIFAR-10 데이터셋을 활용한 CNN 모델 구축

### 📌 주요 코드
```python
# CIFAR-10 이미지와 라벨을 학습/테스트 세트로 불러옴
# 총 60,000개의 32x32 컬러 이미지 (10개 클래스)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 데이터 전처리
# 이미지 픽셀값을 0~1 사이로 정규화
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
# 정답 라벨을 one-hot 인코딩
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# CNN 모델 구성
# 합성곱(Conv2D) → 풀링(MaxPooling2D) → 완전연결(Dense) 구조
# 마지막 레이어에서 10개 클래스 확률 출력 (softmax)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 모델 컴파일
# 최적화 알고리즘: Adam / 손실 함수: 다중 클래스용 categorical crossentropy / 평가 지표: 정확도
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 훈련
# 10번 반복 학습 (에폭) / 배치 크기 64 / 훈련 데이터의 10%를 검증용으로 사용
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

# 테스트 데이터로 성능 측정, 정확도 출력
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"테스트 정확도: {test_acc:.4f}")

# 학습 정확도 및 검증 정확도를 에폭(epoch)별로 시각화
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

# 테스트 이미지 일부에 대해 예측 수행(3개)
# 실제 라벨과 예측 라벨을 이미지와 함께 시각화
sample_images = x_test[:3]
sample_labels = y_test[:3]
predictions = model.predict(sample_images)

for i in range(3):
    plt.imshow(sample_images[i])
    plt.title(f"prediction: {class_names[np.argmax(predictions[i])]} | actual: {class_names[np.argmax(sample_labels[i])]}")
    plt.axis('off')
    plt.show()
```

### ✅ 구현 결과
<img width="476" alt="image" src="https://github.com/user-attachments/assets/c2915b4b-efac-4dda-981a-85deb2488fb2" />
<img width="113" alt="image" src="https://github.com/user-attachments/assets/9334cd2f-0afe-4e48-9d9c-a8ece2725f72" />

+ 예측 수행 3개
<img width="476" alt="image" src="https://github.com/user-attachments/assets/b048b5c0-8c25-48e8-9405-d89cdce80f3d" />
<img width="476" alt="image" src="https://github.com/user-attachments/assets/24b7986e-9614-4565-bfce-293d3a647868" />
<img width="473" alt="image" src="https://github.com/user-attachments/assets/11c3f6c3-20ca-48d7-82fb-26c1cbcf4223" /> 

