import io
import matplotlib.pyplot as plt
import numpy as np
import base64
#from flask import app

def make_Image(base_probabilities, depth_probabilities):
    # 클래스 레이블
    skin_types = ['oily', 'normal', 'dry']
    skin_troubles = ['acne', 'wrinkles']

    # Base skin type 확률
    base_labels = [f'{k} (Base)' for k in skin_types]
    base_values = [base_probabilities.get(k, 0) for k in skin_types]

    # Depth skin trouble 확률
    depth_labels = [f'{k} (Depth)' for k in skin_troubles]
    depth_values = [depth_probabilities.get(k, 0) for k in skin_troubles]

    # 막대 그래프의 x축 위치
    x = np.arange(len(skin_types + skin_troubles))
    width = 0.35  # 막대의 너비

    # 그림과 축 설정
    fig, ax = plt.subplots(figsize=(12, 8))

    # Base Skin Type 막대 그래프
    bars1 = ax.bar(x[:len(skin_types)], base_values, width, label='Base Skin Type', color='#1f77b4', edgecolor='black',
                   linewidth=1.2)

    # Depth Skin Type 막대 그래프
    bars2 = ax.bar(x[len(skin_types):], depth_values, width, label='Depth Skin Type', color='#ff7f0e',
                   edgecolor='black', linewidth=1.2)

    # 그래프 스타일링
    ax.set_xlabel('Skin Types and Troubles', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=14, fontweight='bold')
    ax.set_title('Skin Type and Skin Trouble Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(skin_types + skin_troubles, rotation=45, ha='right', fontsize=12)
    ax.legend(fontsize=12)

    # 축 스타일링
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # 그리드 추가
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # 막대 위에 값 추가
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', va='bottom', ha='center', fontsize=10,
                color='black')

    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', va='bottom', ha='center', fontsize=10,
                color='black')

    # 메모리 버퍼에 저장
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    img_str = base64.b64encode(buf.getvalue()).decode()

    #app.logger.info(f"resultImage_str: {img_str} (type: {type(img_str)})")
    return img_str
