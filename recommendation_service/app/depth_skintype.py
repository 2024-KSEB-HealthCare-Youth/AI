import os
import io
from ultralytics import YOLO
from PIL import Image

base_dir = os.path.dirname(os.path.abspath(__file__))
pt_path = os.path.join(base_dir, '../data/best.pt')
image_save_path = os.path.join(base_dir, '../data/test.jpg')  # 이미지를 저장할 경로
model = YOLO(pt_path)

def analyze_depth_skintype(file):
    file.seek(0)

    # 파일 객체로부터 이미지 열기
    img = Image.open(io.BytesIO(file.read()))

    # 이미지 저장
    img.save(image_save_path)

    # 이미지 경로를 사용하여 예측 수행
    results = model(image_save_path)

    # 클래스 이름 리스트 (모델에 정의된 순서대로)
    class_names = ['ACNE', 'dry', 'normal', 'oily', 'WRINKLES']

    # 각 클래스의 초기 확률을 0으로 설정
    probabilities = {'ACNE': 0.0, 'WRINKLES': 0.0}

    detected_classes = set()

    # 첫 번째 결과에서 박스와 신뢰도 점수 가져오기
    boxes = results[0].boxes
    confidences = boxes.conf
    class_ids = boxes.cls

    if len(boxes) > 0:
        for cls, conf in zip(class_ids, confidences):
            class_id = int(cls)
            class_name = class_names[class_id]

            # 'acne'와 'wrinkles'만 처리
            if class_name == 'ACNE':
                detected_classes.add('ACNE')
                probabilities['ACNE'] = max(probabilities['ACNE'], conf.item()+0.35)
            elif class_name == 'WRINKLES':
                detected_classes.add('WRINKLES')
                probabilities['WRINKLES'] = max(probabilities['WRINKLES'], conf.item()+0.35)
    # 저장된 이미지 삭제
    os.remove(image_save_path)
    return list(detected_classes), probabilities
