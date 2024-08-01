import json
from base_skintype import analyze_base_skintype
from depth_skintype import analyze_depth_skintype
from CBR import get_recommendations
from nutr_recommendation import get_recommended_nutrs
from collaborative_filtering import get_recommendations_collabo


def check_serializability(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False

# 예시 사용법
print(check_serializability(recommendations))
print(check_serializability(recommendations_collabo))
print(check_serializability(nutrs_recommendations))
