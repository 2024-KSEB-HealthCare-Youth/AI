import os
import numpy as np
from flask import Flask, request, jsonify
from flask_restx import Resource, Namespace
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BST = Namespace('BST')
# 모델 로드
model = load_model('recommendation-service/data/custom_mobilenetv2_model.h5')

# 클래스 레이블 정의 (데이터셋에 맞게 수정)
class_labels = ['oily', 'normal', 'dry']

@BST.route('')
class base_skintype(Resource):
    def post(self):
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            # 이미지 전처리
            img = load_img(file, target_size=(224, 224))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            # 예측
            predictions = model.predict(img)
            predicted_class = class_labels[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])

            return jsonify({
                'predicted_class': predicted_class,
                'confidence': float(confidence)
            })