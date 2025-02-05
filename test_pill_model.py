import tensorflow as tf
import numpy as np
import os

# 1. 모델 로드
model = tf.keras.models.load_model("best_model.keras")

# 2. 클래스 이름 로드 (학습 시 저장된 class_names 필요)
# 학습에 사용된 class_names 리스트를 동일하게 로드
class_names = [
    "K-019461",
    "K-019469",
    "K-019553",
    "K-019699",
    "K-019700",
    "K-019790",
    "K-019843",
    "K-019846",
    "K-019860",
    "K-019861",
    "K-019867",
    "K-019904",
    "K-019963",
    "K-020006",
    "K-020014",
    "K-020251",
    "K-020330",
    "K-020378",
    "K-020379",
    "K-020401",
    "K-020582",
    "K-020665",
    "K-020769",
    "K-020805",
    "K-020834",
    "K-020877",
    "K-020966",
    "K-020967",
    "K-021118",
    "K-021426",
    "K-021617",
    "K-021904",
    "K-021930",
    "K-022095",
    "K-022176",
    "K-022183",
    "K-022318",
    "K-022319",
    "K-022362",
    "K-022371",
    "K-022377",
    "K-022465",
    "K-022604",
    "K-022644",
    "K-022703",
    "K-022704",
    "K-022705",
    "K-022706",
    "K-022712",
    "K-022713"
]

# 3. 이미지 전처리 함수
def preprocess_image(image_path):
    """테스트 이미지를 모델 입력 형식에 맞게 전처리합니다."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)  # PNG 이미지 디코딩
    image = tf.image.resize(image, (224, 224))  # 모델 입력 크기로 조정
    image = tf.keras.applications.efficientnet.preprocess_input(image)  # EfficientNet 전처리
    return tf.expand_dims(image, axis=0)  # 배치 차원 추가

# 4. 테스트 함수
def predict_image(image_path):
    """이미지 경로를 받아 클래스 예측 결과를 반환합니다."""
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]  # 가장 높은 확률의 클래스 인덱스
    confidence = np.max(predictions)  # 가장 높은 확률 값

    print(f"Predicted Class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.2f}")

    return class_names[predicted_class], confidence

# 5. 테스트 이미지 경로
# 테스트할 이미지 경로 지정
TEST_IMAGE_PATH = "./166.약품식별_인공지능_개발을_위한_경구약제_이미지_데이터/01.데이터/1.Training/원천데이터/단일경구약제_5000종/TS_10_단일/K-020665/K-020665_0_2_0_2_70_320_200.png"  # 실제 이미지 경로로 대체

# 6. 예측 실행
if os.path.exists(TEST_IMAGE_PATH):
    predicted_class, confidence = predict_image(TEST_IMAGE_PATH)
    print(f"테스트 이미지 결과: {predicted_class} ({confidence * 100:.2f}%)")
else:
    print("테스트 이미지 경로가 존재하지 않습니다. 경로를 확인하세요.")
