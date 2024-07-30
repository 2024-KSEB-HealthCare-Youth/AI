from surprise import Dataset, Reader, KNNBasic
import pandas as pd

# 가상의 데이터셋 생성
data = {
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
    'item_id': [1, 4, 10, 20, 50, 7, 100, 52, 33, 57, 83, 92],
    'rating': [3, 2, 2, 2, 5, 3, 2, 4, 5, 4, 4, 1],
    'skin_type': ['dry', 'dry', 'oily', 'oily', 'normal', 'normal', 'dry', 'dry', 'normal', 'normal', 'oily', 'oily']
}
df = pd.DataFrame(data)

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


def get_recommendations_for_user(user_id, n=3):
    """특정 사용자에 대한 상위 N개 추천 아이템을 반환합니다."""
    # 사용자가 아직 평가하지 않은 모든 아이템 가져오기
    items_to_predict = df[~df['item_id'].isin(df[df['user_id'] == user_id]['item_id'])]['item_id'].unique()

    # 각 아이템에 대한 예측 평점 계산
    predictions = [algo.predict(user_id, item_id) for item_id in items_to_predict]

    # 예측 평점이 높은 순으로 정렬
    top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]

    return [(pred.iid, pred.est) for pred in top_predictions]


# 사용자 입력 받기
user_id = int(input("추천을 받고 싶은 사용자 ID를 입력하세요: "))

# 추천 아이템 가져오기
recommendations = get_recommendations_for_user(user_id)

# 결과 출력
print(f"사용자 {user_id}에 대한 추천 아이템:")
for iid, est in recommendations:
    print(f"  아이템 {iid}, 예측 평점: {est:.2f}")
