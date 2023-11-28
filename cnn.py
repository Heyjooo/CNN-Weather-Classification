import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import shutil
import numpy as np
import os

def copy_file(src, dest, file):
    shutil.copy(os.path.join(src, file), os.path.join(dest, file))

def process_files(origin, dest_train, dest_val, dest_test, file):
    random_number = np.random.randint(99) + 1
    if random_number <= 80:
        copy_file(origin, dest_train, file)
    elif random_number <= 90:
        copy_file(origin, dest_val, file)
    else:
        copy_file(origin, dest_test, file)

types = ['foggy', 'lightning', 'rainy', 'snowy', 'sunny']
base_path = '/Users/byunheejoo/g/CNN/Data/' # 원본데이터
dest_base = '/Users/byunheejoo/PycharmProjects/CNN/Dataset/' # 분류한데이터

for t in types:
    origin = os.path.join(base_path, t)
    dest_train = os.path.join(dest_base, 'train', t)
    dest_val = os.path.join(dest_base, 'validation', t)
    dest_test = os.path.join(dest_base, 'test', t)

    for root, dirs, files in os.walk(origin):
        for file in files:
            process_files(origin, dest_train, dest_val, dest_test, file) #랜덤으로 데이터 분류

# GPU 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 데이터셋 폴더 경로
dataset_dir = dest_base

# 이미지 데이터 전처리
datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen_valid = ImageDataGenerator(rescale=1./255)

# 훈련 데이터 생성
train_generator = datagen_train.flow_from_directory(
    os.path.join(dataset_dir, 'train'),
    target_size=(256, 256),
    batch_size=64,
    class_mode='categorical'
)

# 검증 데이터 생성
validation_generator = datagen_valid.flow_from_directory(
    os.path.join(dataset_dir, 'validation'),
    target_size=(256, 256),
    batch_size=64,
    class_mode='categorical'
)

# CNN 모델
model = Sequential()

# 첫 번째 Convolutional Layer
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(256, 256, 3), activation='relu'))
# 첫 번째 Pooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# 두 번째 Convolutional Layer
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# 두 번째 Pooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# 세 번째 Convolutional Layer
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# 세 번째 Pooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# 네 번째 Convolutional Layer
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
# 네 번째 Pooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten Layer
model.add(Flatten())

# 다섯 번째 Dense Layer (Fully Connected Layer)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))

# 출력 Dense Layer
model.add(Dense(5, activation='softmax'))  # 클래스 5개

# 모델 컴파일 및 요약
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

# 모델 훈련
history = model.fit(
    train_generator,
    epochs=100,
    verbose=2,
    validation_data=validation_generator
)

# Loss & Accuracy 그래프
fig, ax = plt.subplots(2, 1, figsize=(12, 8))

ax[0].plot(history.history['loss'], label='Train Loss')
ax[0].plot(history.history['val_loss'], label='Validation Loss')
ax[0].set_title('Model Loss')
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Epoch')
ax[0].legend()

ax[1].plot(history.history['accuracy'], label='Train Accuracy')
ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
ax[1].set_title('Model Accuracy')
ax[1].set_ylabel('Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].legend()

plt.tight_layout()
plt.savefig('Loss_Accuracy.png')
plt.show()

# 테스트 데이터 생성
datagen_test = ImageDataGenerator(rescale=1./255)
test_generator = datagen_test.flow_from_directory(
    os.path.join(dataset_dir, 'test'),
    target_size=(256, 256),
    batch_size=64,
    class_mode='categorical'
)

# 테스트 데이터로 평가
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

