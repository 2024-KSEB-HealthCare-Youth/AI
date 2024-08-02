import pandas as pd
import os
#from flask import app


def get_recommended_nutrs(base_skin_type, depth_skin_type, max_recommendations=3):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, '../data/nutrients.csv')

    # CSV 파일 읽기
    df = pd.read_csv(file_path)

    # 필터링 조건 정의
    base_filter = df['type'] == base_skin_type
    depth_filters = [df['type'] == dt for dt in depth_skin_type]

    # 제품 리스트 초기화
    recommended_products = []

    # depth_skin_type이 비어 있는 경우
    if not depth_skin_type:
        # base_skin_type과 일치하는 제품을 랜덤으로 섞어 최대 3개 추출
        base_products = df[base_filter].sample(n=min(max_recommendations, len(df[base_filter])), random_state=None)
        recommended_products = base_products.to_dict(orient='records')

    else:
        # base_skin_type과 일치하는 제품을 랜덤으로 1개 추출
        base_products = df[base_filter].sample(n=1, random_state=None).to_dict(orient='records')

        # depth_skin_type과 일치하는 제품을 랜덤으로 추출
        depth_products = []
        for depth_filter in depth_filters:
            depth_products.extend(df[depth_filter].sample(n=1, random_state=None).to_dict(orient='records'))

        # 합쳐서 총 3개까지 반환
        recommended_products.extend(base_products)
        recommended_products.extend(depth_products)
        recommended_products = recommended_products[:max_recommendations]

    #app.logger.info(f"nutrs_recommendations: {recommended_products} (type: {type(recommended_products)})")

    return recommended_products
