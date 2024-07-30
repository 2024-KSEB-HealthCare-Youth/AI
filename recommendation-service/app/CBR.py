import pandas as pd

# CSV 파일에서 데이터프레임 읽기
file_path = 'recommendation-service/data/title_resultType_img.csv'
product_df = pd.read_csv(file_path)

def get_recommendations(base_result):
    user_profile = {
        'skinType': base_result,
        'skinTrouble': ['skin_trouble2']
    }

    recommended_items = content_based_recommendation(product_df, user_profile, top_n=3)

    return recommended_items


def content_based_recommendation(product_df, user_profile, top_n=3):
    product_df['score'] = product_df.apply(
        lambda x: (x['skin_type_result'] == user_profile['skinType']) +
                  sum(trouble in user_profile['skinTrouble'] for trouble in x['skin_trouble_result'].split(',')), axis=1)

    recommended_products = product_df.sort_values(by='score', ascending=False).head(top_n)
    recommend_items = recommended_products[['title', 'imgurl']].to_dict(orient='records')
    return recommend_items