import os
import matplotlib
import json
matplotlib.use('Agg')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from flask import Flask, request, jsonify, Response
from flask_restx import Resource, Api, Namespace, reqparse
from flask_cors import CORS
from base_skintype import analyze_base_skintype
from depth_skintype import analyze_depth_skintype
from CBR import get_recommendations
from nutr_recommendation import get_recommended_nutrs
from collaborative_filtering import get_recommendations_collabo

app = Flask(__name__)
api = Api(app)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024

SkinAnalysis = Namespace('SkinAnalysis')
parser = reqparse.RequestParser()
parser.add_argument('file', location='files', type='file', required=True, help='File to upload')

@SkinAnalysis.route('')
class SkinAnalysisResource(Resource):
    @SkinAnalysis.expect(parser)
    def post(self):
        file = request.files.get('file')

        if not file:
            return jsonify({'error': 'No file provided'}), 400

        try:
            # 파일 정보 출력 (디버그 목적)
            print(f"Received file: {file.filename}, Size: {len(file.read())} bytes")
            file.seek(0)  # 파일 포인터를 처음으로 되돌림

            # Base Skin Type 분석
            base_skin_type, base_probabilities = analyze_base_skintype(file)

            # 파일 포인터를 처음으로 되돌림
            file.seek(0)

            # Depth Skin Type 분석
            depth_skin_type, depth_probabilities = analyze_depth_skintype(file)

            # 두 개의 확률 딕셔너리를 합침
            result_probabilities = {**base_probabilities, **depth_probabilities}

            # CBR 추천
            recommendations = get_recommendations(base_skin_type, depth_skin_type)

            # Collaborative Filtering 추천
            recommendations_collabo = get_recommendations_collabo(base_skin_type)

            # Nutritional 추천
            nutrs_recommendations = get_recommended_nutrs(base_skin_type, depth_skin_type)

            # 결과 집합 초기화
            cosNames = []
            cosPaths = []

            # CBR 추천 추가
            for rec in recommendations_collabo:
                cosNames.append(rec['title'])
                cosPaths.append(rec['imgurl'])

            # Collaborative Filtering 추천 추가
            added_count = 0
            for rec in recommendations:
                if added_count >= 3:  # 최대 3개 항목을 추가
                    break
                if rec['title'] not in cosNames:  # 중복 제거
                    cosNames.append(rec['title'])
                    cosPaths.append(rec['imgurl'])
                    added_count += 1

            # Nutritional 추천 추가
            nutrNames = [nutr['title'] for nutr in nutrs_recommendations]
            nutrPaths = [nutr['url'] for nutr in nutrs_recommendations]

            response = {
                'cosNames': cosNames,
                'cosPaths': cosPaths,
                'nutrNames': nutrNames,
                'nutrPaths': nutrPaths,
                'simpleSkin': base_skin_type,
                'expertSkin': depth_skin_type,
                'probabilities': result_probabilities
            }

            return jsonify(response)
        except Exception as e:
            # 예외 처리 (보안 고려 필요)
            return jsonify({'error': str(e)}), 500

api.add_namespace(SkinAnalysis, '/upload')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # debug 필요하면 debug=True 넣어주기 (실제 서버에서는 안넣는 것이 좋음)
