import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import io


# 모델 로드
model = load_model('recommendation-service/data/custom_mobilenetv2_model.h5')

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
    predicted_class = class_labels[np.argmax(predictions[0])]

    # 스킨 타입 매핑
    if predicted_class == "oily":
        skin_type = "skin_type1"
    elif predicted_class == "normal":
        skin_type = "skin_type2"
    else:
        skin_type = "skin_type3"

    return skin_type
