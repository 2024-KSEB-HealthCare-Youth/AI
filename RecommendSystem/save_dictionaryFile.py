import pandas as pd

# 딕셔너리 예시
data = [
    {'name': 'Alice', 'age': 30, 'city': 'New York'},
    {'name': 'Bob', 'age': 25, 'city': 'Los Angeles'},
    {'name': 'Charlie', 'age': 35, 'city': 'Chicago'}
]

# 딕셔너리를 JSON 파일로 저장
def save_dict_to_json(data, file_path):
    df = pd.DataFrame(data)
    df.to_json(file_path, orient='records', indent=4)

# JSON 파일에서 딕셔너리를 불러오기
def load_dict_from_json(file_path):
    df = pd.read_json(file_path, orient='records')
    return df.to_dict(orient='records')

# 파일 경로를 지정해 주세요
file_path = 'data.json'

# 딕셔너리를 파일로 저장
save_dict_to_json(data, file_path)

# 파일에서 딕셔너리를 불러오기
loaded_data = load_dict_from_json(file_path)
print(loaded_data)
