import os
import tensorflow as tf

# 1. 데이터 로드 및 클래스 매핑
def load_data(base_dir):
    """
    폴더 구조를 기반으로 데이터와 라벨을 로드합니다.
    - base_dir: 이미지 데이터셋의 최상위 디렉토리
    """
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(base_dir))  # 클래스 이름 정렬 (하위 폴더명)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_file in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, img_file))
            labels.append(class_to_idx[class_name])

    return image_paths, labels, class_names


# 2. 데이터셋 경로 설정
BASE_DIR = "./166.약품식별_인공지능_개발을_위한_경구약제_이미지_데이터/01.데이터/1.Training/원천데이터/단일경구약제_5000종/TS_10_단일"
image_paths, labels, class_names = load_data(BASE_DIR)

# 3. 데이터셋 분리
from sklearn.model_selection import train_test_split

# 데이터를 학습용과 검증용으로 분리
train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42
)

# 4. 데이터셋 생성 및 전처리
def preprocess_image(image_path, label):
    """이미지를 전처리하고 라벨과 함께 반환합니다."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label

# 학습 데이터셋 생성
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
train_dataset = (
    train_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(buffer_size=1024)
    .batch(32)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# 검증 데이터셋 생성
validation_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_labels))
validation_dataset = (
    validation_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# 5. 데이터 증강
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

def augment_data(image, label):
    image = data_augmentation(image)
    return image, label

train_dataset = train_dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)

# 6. 모델 생성
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)
base_model.trainable = True
for layer in base_model.layers[:100]:  # 일부 레이어 고정
    layer.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation="softmax"),
])

# 7. 학습률 스케줄링 및 컴파일
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9,
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 8. 학습
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,  # 검증 데이터 전달
    epochs=15,
    callbacks=[early_stopping, checkpoint_cb]
)

# 9. 모델 저장
model.save("pill_model.h5")
