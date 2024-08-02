from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
pt_path = os.path.join(base_dir, '../data/best.pt')
image_save_path = os.path.join(base_dir, '../data/uploaded_image.jpg')  # 이미지를 저장할 경로


def analyze_depth_skintype(file):
    file.seek(0)
    print('depth 도달!')
    # YOLO 모델 불러오기
    model = YOLO(pt_path)
    print('모델 로딩 성공!')

    # 파일 객체로부터 이미지 열기
    img = Image.open(io.BytesIO(file.read()))
    print('이미지 오픈 성공!')
    img = img.resize((256, 256))

    # 이미지를 저장
    img.save(image_save_path)
    print('이미지 저장 성공!')

    # 이미지 경로를 사용하여 예측 수행
    print('수행 시작!')
    results = model(image_save_path)
    print('수행 끝!')

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

    # 저장된 이미지를 삭제
    os.remove(image_save_path)
    print('이미지 삭제 성공!')

    return list(detected_conditions), probabilities

# 예제 사용법
# with open('path_to_image.jpg', 'rb') as image_file:
#     detected_conditions, probabilities = analyze_depth_skintype(image_file)
#     print(f"Detected Conditions: {detected_conditions}")
#     print(f"Probabilities: {probabilities}")
