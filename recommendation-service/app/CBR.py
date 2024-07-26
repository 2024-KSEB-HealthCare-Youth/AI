# recommendation.py
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_restx import Namespace, Resource

CBR = Namespace('CBR', description='Content-based recommendation operations')

# CSV 파일에서 데이터프레임 읽기
file_path = 'recommendation-service/data/title_resultType_img.csv'
product_df = pd.read_csv(file_path)

@CBR.route('')
class ContentBasedRecommendation(Resource):
    def post(self):
        # 파일이 요청에 포함되어야 합니다.
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        #file = request.files['file']
        file = "recommendation-service/data/img.jpg"
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # /predict 엔드포인트 호출
        predict_url = 'http://127.0.0.1:5000/BST/predict'
        files = {'file': file}
        response = requests.post(predict_url, files=files)
        predict_data = response.json()

        if response.status_code == 200:
            predicted_class = predict_data['predicted_class']

            # 사용자 프로필을 업데이트
            user_profile = {
                'skinType': predicted_class,
                'skinTrouble': ['skin_trouble2']
            }

            # 추천 아이템 생성
            recommended_items = self.content_based_recommendation(product_df, user_profile, top_n=3)

            flutter_server_url = "http://localhost:3000/receive_recommendation"  # Flutter 서버 엔드포인트
            result_data = {
                'predicted_class': predicted_class,
                'recommended_items': recommended_items
            }
            '''
            try:
                flutter_response = requests.post(flutter_server_url, json=result_data)
                if flutter_response.status_code == 200:
                    return jsonify({'message': 'Data sent to Flutter server successfully'}), 200
                else:
                    return jsonify({
                                       'error': f'Failed to send data to Flutter server. Status code: {flutter_response.status_code}'}), 500
            except requests.RequestException as e:
                return jsonify({'error': f'Error sending data to Flutter server: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Failed to get prediction'}), 500
            
            '''
            return jsonify({
                'predicted_class': predicted_class,
                'recommended_items': recommended_items
            })
        else:
            return jsonify({'error': 'Failed to get prediction'}), 500



    def content_based_recommendation(self, product_df, user_profile, top_n=3):
        product_df['score'] = product_df.apply(
            lambda x: (x['skin_type_result'] == user_profile['skinType']) +
                      (x['skin_trouble_result'] in user_profile['skinTrouble']), axis=1)

        recommended_products = product_df.sort_values(by='score', ascending=False).head(top_n)
        recommended_items = recommended_products[['title', 'imgurl']].to_dict(orient='records')
        return recommended_items
