from flask import Flask, request, jsonify
from flask_restx import Resource, Api, Namespace
from flask_cors import CORS
from base_skintype import analyze_base_skintype
from CBR import get_recommendations

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
            base_result = analyze_base_skintype(file)

            # 파일 포인터를 처음으로 되돌립니다
            file.seek(0)

            # Depth Skin Type 분석
            #depth_result = analyze_depth_skintype(file)

            # CBR 추천
            recommendations = get_recommendations(base_result) #여기에 depth_result도 받을 수 있게 나중에

            # 'depth_skin_type': depth_result, 이 코드 return에 넣자
            return jsonify({
                'base_skin_type': base_result,
                'recommendations': recommendations
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

api.add_namespace(SkinAnalysis, '/upload')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) #debug 필요하면 debug=True 넣어주기 (실제 서버에서는 안넣는 것이 좋음)
