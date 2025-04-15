# 1. 필요한 라이브러리 불러오기
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# 2. CIFAR-10 데이터셋 불러오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 3. 데이터 전처리
# 이미지 픽셀값 0~255 → 0~1로 정규화
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 정답 라벨을 One-hot 인코딩
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 4. CNN 모델 구성
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10개 클래스 출력
])

# 5. 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. 모델 훈련
history = model.fit(x_train, y_train, epochs=10, batch_size=64,
                    validation_split=0.1)

# 7. 성능 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"테스트 정확도: {test_acc:.4f}")

# 8. 정확도 시각화
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('CIFAR-10 Classification Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# 9. 테스트 이미지 예측 예시
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 무작위 테스트 이미지 3장 예측
sample_images = x_test[:3]
sample_labels = y_test[:3]
predictions = model.predict(sample_images)

for i in range(3):
    plt.imshow(sample_images[i])
    plt.title(f"prediction: {class_names[np.argmax(predictions[i])]} | actual: {class_names[np.argmax(sample_labels[i])]}")
    plt.axis('off')
    plt.show()
