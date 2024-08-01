from flask import Flask, request, jsonify
from flask_restx import Resource, Api, Namespace
from flask_cors import CORS
from base_skintype import analyze_base_skintype
from depth_skintype import analyze_depth_skintype
from CBR import get_recommendations
from nutr_recommendation import get_recommended_nutrs
from collaborative_filtering import get_recommendations_collabo
from resultImage import make_Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)
api = Api(app)
CORS(app)

SkinAnalysis = Namespace('SkinAnalysis')

@SkinAnalysis.route('')
class SkinAnalysisResource(Resource):
    def post(self):
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file provided'}), 400

        try:
            # Base Skin Type 분석
            base_skin_type, base_probabilities = analyze_base_skintype(file)

            # 파일 포인터를 처음으로 되돌립니다
            file.seek(0)

            # Depth Skin Type 분석
            depth_skin_type, depth_probabilities = analyze_depth_skintype(file)

            # CBR 추천
            recommendations = get_recommendations(base_skin_type, depth_skin_type)
            
            #collaborative-filtering 추천
            recommendations_collabo = get_recommendations_collabo(base_skin_type)

            # nutr 추천
            nutrs_recommendations = get_recommended_nutrs(base_skin_type, depth_skin_type)

            #resultImage 생성
            resultImage_str = make_Image(base_probabilities, depth_probabilities)

            # 결과 집합 초기화
            cosNames = []
            cosPaths = []

            # CBR 추천 추가
            for rec in recommendations:
                cosNames.append(rec['title'])
                cosPaths.append(rec['imgurl'])

                # Collaborative Filtering 추천 추가
                added_count = 0
                for rec in recommendations_collabo:
                    if added_count >= 3:  # 최대 3개 항목을 추가
                        break
                    if rec['title'] not in cosNames:  # 중복 제거
                        cosNames.append(rec['title'])
                        cosPaths.append(rec['imgurl'])
                        added_count += 1

                    # nutr 추천
                    nutrNames = [nutr['title'] for nutr in nutrs_recommendations]
                    nutrPaths = [nutr['url'] for nutr in nutrs_recommendations]

            response = {
                'cosNames': cosNames,
                'cosPaths': cosPaths,
                'nutrNames': nutrNames,
                'nutrPaths': nutrPaths,
                'simpleSkin': base_skin_type,
                'expertSkin': depth_skin_type,  # 여기에 실제 expertSkin 정보를 넣으세요
                'resultImage': resultImage_str  # 여기에 실제 resultPath를 설정하세요
            }

            return jsonify(response)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

api.add_namespace(SkinAnalysis, '/upload')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) #debug 필요하면 debug=True 넣어주기 (실제 서버에서는 안넣는 것이 좋음)
