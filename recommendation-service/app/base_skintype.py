import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import io
import os

# 모델 로드
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '../data/custom_mobilenetv2_model.h5')
model = load_model(model_path)

# 클래스 레이블 정의 (데이터셋에 맞게 수정)
class_labels = ['oily', 'normal', 'dry']

def analyze_base_skintype(file):
    img = Image.open(io.BytesIO(file.read()))

    # 이미지 전처리
    img = img.resize((224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # 예측
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_class_index]

    # 각 클래스에 대한 확률 매핑
    probabilities = {class_labels[i]: float(predictions[0][i]) for i in range(len(class_labels))}

    return predicted_class, probabilities

# Example usage:
# with open('path_to_image.jpg', 'rb') as image_file:
#     skin_type, predicted_probability, probabilities = analyze_base_skintype(image_file)
#     print(f"Skin Type: {skin_type}, Probability: {predicted_probability}")
#     print(f"All Probabilities: {probabilities}")
