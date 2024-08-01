import pandas as pd
import os

# CSV 파일에서 데이터프레임 읽기
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, '../data/title_resultType_img.csv')
product_df = pd.read_csv(file_path)
def get_recommendations(base_skin_type, depth_skin_type):

    # 스킨 타입 매핑
    if base_skin_type == "oily":
        skin_type = "skin_type1"
    elif base_skin_type == "combination":
        skin_type = "skin_type2"
    else:
        skin_type = "skin_type3"

    # depth_skin_type을 변환
    skin_trouble_mapping = {
        'acne': 'skin_trouble2',
        'wrinkles': 'skin_trouble3'
    }
    skin_trouble = [skin_trouble_mapping.get(condition, condition) for condition in depth_skin_type]

    user_profile = {
        'skinType': skin_type,
        'skinTrouble': skin_trouble
    }

    recommended_items = content_based_recommendation(product_df, user_profile, top_n=3)

    return recommended_items


def content_based_recommendation(product_df, user_profile, top_n=3):
    product_df['score'] = product_df.apply(
        lambda x: (x['skin_type_result'] == user_profile['skinType']) +
                  sum(trouble in user_profile['skinTrouble'] for trouble in x['skin_trouble_result'].split(',')), axis=1)

    # 스코어에 따라 정렬 후 무작위로 섞기
    recommended_products = product_df.sort_values(by='score', ascending=False)
    recommended_products = recommended_products.head(2 * top_n)  # 상위 2 * top_n개 아이템 중 무작위 선택
    recommended_products = recommended_products.sample(frac=1).head(top_n)  # 무작위로 섞은 후 상위 top_n개 선택

    recommend_items = recommended_products[['title', 'imgurl']].to_dict(orient='records')
    return recommend_items