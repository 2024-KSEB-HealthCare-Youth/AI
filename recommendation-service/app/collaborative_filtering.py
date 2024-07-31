from surprise import Dataset, Reader, KNNBasic
import pandas as pd
import os
import numpy as np

# CSV 파일 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
data_csv_path = os.path.join(base_dir, '../data/data.csv')
item_data_csv_path = os.path.join(base_dir, '../data/item_data.csv')

# 데이터와 아이템 정보가 담긴 CSV 파일을 로드합니다
df = pd.read_csv(data_csv_path)
item_df = pd.read_csv(item_data_csv_path)

# Surprise 데이터셋 생성
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

# 전체 데이터셋을 학습 세트로 사용
trainset = dataset.build_full_trainset()

# 사용자 정의 유사도 함수
def skin_type_similarity(u1, u2):
    user1_skin = df[df['user_id'] == u1['user_id']]['skin_type'].iloc[0]
    user2_skin = df[df['user_id'] == u2['user_id']]['skin_type'].iloc[0]
    return 1 if user1_skin == user2_skin else 0

# KNN 모델 생성 및 학습
sim_options = {
    'name': 'skin_type_similarity',
    'user_based': True  # 사용자 기반 협업 필터링
}
algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)

def get_recommendations_collabo(skin_type, n=3):
    """특정 피부 타입에 대한 상위 N개 추천 아이템을 반환합니다."""
    # 해당 피부 타입을 가진 사용자들이 평가한 아이템
    skin_type_users = df[df['skin_type'] == skin_type]['user_id'].unique()
    skin_type_items = df[df['user_id'].isin(skin_type_users)]['item_id'].unique()

    # 사용자가 아직 평가하지 않은 아이템만 필터링
    items_to_predict = df[~df['item_id'].isin(skin_type_items)]['item_id'].unique()

    # 각 아이템에 대한 예측 평점 계산
    predictions = [algo.predict(user_id, item_id) for item_id in items_to_predict for user_id in skin_type_users]

    # 예측 평점이 높은 순으로 정렬
    top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

    # 상위 2 * n개의 아이템 중 무작위로 n개 선택
    if len(top_predictions) > 2 * n:
        top_predictions = top_predictions[:2 * n]
        np.random.shuffle(top_predictions)
        top_predictions = top_predictions[:n]
    else:
        np.random.shuffle(top_predictions)
        top_predictions = top_predictions[:n]

    # 추천 아이템의 ID와 예측 평점을 DataFrame으로 변환
    top_items = [(pred.iid, pred.est) for pred in top_predictions]
    top_items_df = pd.DataFrame(top_items, columns=['item_id', 'est'])

    # 추천 아이템 정보가 담긴 CSV 파일에서 title과 imgurl 조회
    recommended_products = item_df[item_df['Item Number'].isin(top_items_df['item_id'])]
    recommend_items = recommended_products[['title', 'imgurl']].to_dict(orient='records')

    return recommend_items
