from ultralytics import YOLO
from PIL import Image
import io
#from flask import app

def analyze_depth_skintype(file):
    # YOLO 모델 불러오기
    model = YOLO('best.pt')

    # 파일 객체로부터 이미지 열기
    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((256, 256))

    # 이미지에 대한 예측 수행
    results = model(img)

    # 클래스 이름 리스트 (모델에 정의된 순서대로)
    class_names = ['acne', 'dry', 'normal', 'oily', 'wrinkles']

    # 각 클래스의 초기 확률을 0으로 설정
    probabilities = {'acne': 0.0, 'wrinkles': 0.0}

    detected_conditions = set()

    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            class_name = class_names[cls]
            confidence = box.conf  # YOLOv5의 박스에서 확률 가져오기
            # 'wrinkle'이나 'acne'인 경우에만 처리
            if class_name in ['wrinkles', 'acne']:
                if class_name == 'acne':
                    detected_conditions.add('acne')
                    probabilities['acne'] = max(probabilities['acne'], confidence)
                elif class_name == 'wrinkles':
                    detected_conditions.add('wrinkles')
                    probabilities['wrinkles'] = max(probabilities['wrinkles'], confidence)

    #app.logger.info(f"depth_skin_type: {list(detected_conditions)} (type: {type(list(detected_conditions))})")
    #app.logger.info(f"depth_probabilities: {probabilities} (type: {type(probabilities)})")

    return list(detected_conditions), probabilities

# 예제 사용법
# with open('path_to_image.jpg', 'rb') as image_file:
#     detected_conditions, probabilities = analyze_depth_skintype(image_file)
#     print(f"Detected Conditions: {detected_conditions}")
#     print(f"Probabilities: {probabilities}")
