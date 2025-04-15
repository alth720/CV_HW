# 1. 필요한 라이브러리 불러오기
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt  # 시각화를 위한 matplotlib 추가

# 2. MNIST 데이터셋 불러오기
# 데이터를 train/test 세트로 자동 분리해줌
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 3. 데이터 전처리
# 28x28 이미지를 0~255 범위에서 0~1 사이로 정규화
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 신경망에 입력하려면 (28, 28) 이미지를 1차원으로 펼쳐야 함 → (784,)
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# 라벨(정답)을 one-hot 인코딩
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 4. 신경망 모델 구성
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),  # 은닉층
    layers.Dense(64, activation='relu'),                       # 은닉층
    layers.Dense(10, activation='softmax')                     # 출력층
])

# 5. 모델 컴파일 (옵티마이저, 손실함수, 평가 지표 설정)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# 6. 모델 훈련 (History 객체 받아오기)
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# 7. 테스트 세트로 정확도 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"테스트 정확도: {test_acc:.4f}")

# ✅ 8. 정확도 시각화
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
