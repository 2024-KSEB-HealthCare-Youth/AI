import pandas as pd

# CSV 파일에서 데이터프레임 읽기
file_path = '../OliveYoungCrawling/resultSkinType.csv'
product_df = pd.read_csv(file_path)

# 새로운 사용자의 기본 정보
new_user_profile = {
    'skinType': 'skin_type2',
    'skinTrouble': ['skin_trouble2']
}

def content_based_recommendation(product_df, user_profile, top_n=3):
    # 사용자 프로필과 제품의 유사도를 계산 (간단한 예: 카테고리와 브랜드가 일치하는 제품을 추천)
    product_df['score'] = product_df.apply(
        lambda x: (x['skin_type_result'] == user_profile['skinType']) +
                  (x['skin_trouble_result'] in user_profile['skinTrouble']), axis=1)

    # 스코어가 높은 상위 N개의 제품 추천
    recommended_products = product_df.sort_values(by='score', ascending=False).head(top_n)
    return recommended_products['title'].tolist()

# 새로운 사용자에게 Content-based 추천
recommended_items = content_based_recommendation(product_df, new_user_profile, top_n=3)
print(f"추천 아이템: {recommended_items}")
