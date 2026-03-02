# api.py - 最终修正版 (支持高阶行为特征和批量预测)

from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
import pandas as pd
import numpy as np
import os
import sys
import traceback  # 用于打印详细错误信息
from io import StringIO  # 用于处理批量预测上传的CSV数据

# --- 配置 ---
MODEL_PATH = 'retrain_prediction_model.joblib'
# 请确保您的数据文件路径正确
DATA_DIR = 'data/data_format1'

# 训练模型时使用的特征列 (必须与模型训练时完全一致)
FEATURES = [
    # 基础特征 (6个)
    'user_id', 'merchant_id', 'age_range', 'gender',
    'user_total_action', 'um_total_action',

    # 高阶次数特征 (4个)
    'um_action_0_click', 'um_action_1_addcart', 'um_action_2_buy', 'um_action_3_fav',

    # 高阶比例特征 (4个)
    'um_buy_click_ratio', 'um_fav_click_ratio', 'um_buy_total_ratio', 'um_buy_fav_addcart_ratio'
]

app = Flask(__name__)
CORS(app)  # 开启跨域访问

model = None
USER_INFO_DF = None
USER_LOG_DF = None


# --- 数据加载与特征获取 ---

def load_data():
    """加载模型依赖的数据文件，并进行必要的预处理"""
    global USER_INFO_DF, USER_LOG_DF

    user_info_path = os.path.join(DATA_DIR, 'user_info_format1.csv')
    user_log_path = os.path.join(DATA_DIR, 'user_log_format1.csv')

    try:
        # 1. 加载数据
        USER_INFO_DF = pd.read_csv(user_info_path)
        USER_LOG_DF = pd.read_csv(user_log_path)

        # 2. 清理和重命名 (将 seller_id 重命名为 merchant_id)
        if 'seller_id' in USER_LOG_DF.columns:
            USER_LOG_DF.rename(columns={'seller_id': 'merchant_id'}, inplace=True)
            print("API: 成功重命名 user_log 中的 'seller_id' 为 'merchant_id'。")

        # 3. 统一 ID 列的类型
        USER_INFO_DF['user_id'] = USER_INFO_DF['user_id'].astype(int)
        USER_LOG_DF['user_id'] = USER_LOG_DF['user_id'].astype(int)
        USER_LOG_DF['merchant_id'] = USER_LOG_DF['merchant_id'].astype(int)

        # 4. 处理用户画像的缺失值
        USER_INFO_DF['gender'].fillna(-1, inplace=True)
        USER_INFO_DF['age_range'].fillna(0, inplace=True)

        print("数据加载成功！")

    except FileNotFoundError as e:
        print(f"数据文件加载失败！请检查路径：{e}")
        raise e
    except Exception as e:
        print(f"数据预处理失败: {e}")
        raise e


def load_model():
    """加载模型"""
    global model
    try:
        model = load(MODEL_PATH)
        print("模型加载成功！")
    except Exception as e:
        raise e


def get_features(user_id, merchant_id):
    """根据ID获取所有所需的特征（包括高阶行为比例特征）"""

    feature_dict = {f: 0 for f in FEATURES}
    feature_dict['user_id'] = int(user_id)
    feature_dict['merchant_id'] = int(merchant_id)

    # --- 基础画像特征 ---
    user_data_row = USER_INFO_DF[USER_INFO_DF['user_id'] == user_id]
    if not user_data_row.empty:
        user_data = user_data_row.iloc[0]
        feature_dict['age_range'] = int(user_data.get('age_range', 0))
        feature_dict['gender'] = int(user_data.get('gender', -1))

    # --- 行为特征计算 ---
    user_log_entry = USER_LOG_DF[USER_LOG_DF['user_id'] == user_id]

    user_total_action = len(user_log_entry)
    feature_dict['user_total_action'] = user_total_action

    um_log_entry = user_log_entry[user_log_entry['merchant_id'] == merchant_id]
    um_total_action = len(um_log_entry)
    feature_dict['um_total_action'] = um_total_action

    if um_total_action > 0:
        um_action_counts = um_log_entry.groupby('action_type')['item_id'].count()

        click_count = um_action_counts.get(0, 0)
        addcart_count = um_action_counts.get(1, 0)
        buy_count = um_action_counts.get(2, 0)
        fav_count = um_action_counts.get(3, 0)

        feature_dict['um_action_0_click'] = click_count
        feature_dict['um_action_1_addcart'] = addcart_count
        feature_dict['um_action_2_buy'] = buy_count
        feature_dict['um_action_3_fav'] = fav_count

        epsilon = 1e-6  # 防止除以 0

        feature_dict['um_buy_click_ratio'] = buy_count / (click_count + epsilon)
        feature_dict['um_fav_click_ratio'] = fav_count / (click_count + epsilon)
        feature_dict['um_buy_total_ratio'] = buy_count / (um_total_action + epsilon)
        feature_dict['um_buy_fav_addcart_ratio'] = buy_count / (addcart_count + fav_count + epsilon)

    return [feature_dict[f] for f in FEATURES]


# --- 路由定义 ---

@app.route('/')
def home():
    return "天猫复购预测 API 运行中！请使用前端页面或 /predict 路由进行 POST 请求。"


@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "模型未成功加载，无法进行预测。请检查模型文件是否存在。"}), 500

    try:
        data = request.get_json(force=True)
        user_id = int(data.get('user_id'))
        merchant_id = int(data.get('merchant_id'))

        feature_vector = get_features(user_id, merchant_id)
        input_df = pd.DataFrame([feature_vector], columns=FEATURES)

        prob = model.predict_proba(input_df)[0][1]

        return jsonify({
            "user_id": user_id,
            "merchant_id": merchant_id,
            "repurhase_prob": round(prob, 4)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"预测失败：{str(e)}", "message": "Prediction failed due to internal error."}), 400


@app.route('/bulk_predict', methods=['POST'])
def bulk_predict():
    if not model:
        return jsonify({"error": "模型未成功加载，无法进行批量预测。"}), 500

    try:
        # 1. 获取上传的 CSV 数据
        csv_data = request.data.decode('utf-8')

        # 2. 将 CSV 文本读取为 DataFrame
        input_df = pd.read_csv(StringIO(csv_data))

        # 3. 验证输入列
        required_cols = ['user_id', 'merchant_id']
        if not all(col in input_df.columns for col in required_cols):
            return jsonify({"error": "CSV 文件必须包含 'user_id' 和 'merchant_id' 两列。"}), 400

        # 4. 批量特征工程和预测

        # 使用 apply 函数批量计算特征向量和预测
        def process_row(row):
            user_id = int(row['user_id'])
            merchant_id = int(row['merchant_id'])

            # 运行 get_features 函数，计算所有 16 个特征
            feature_vector = get_features(user_id, merchant_id)

            # 准备模型输入
            input_features = pd.DataFrame([feature_vector], columns=FEATURES)

            # 进行预测
            prob = model.predict_proba(input_features)[0][1]

            return {
                "user_id": user_id,
                "merchant_id": merchant_id,
                "prob": round(prob, 4)
            }

        # 对每一行应用 process_row 函数，并转为列表
        results = input_df.apply(process_row, axis=1).tolist()

        return jsonify(results)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"批量预测失败：{str(e)}", "message": "Bulk prediction failed."}), 400


# --- 启动块 ---
if __name__ == '__main__':
    print("--- 尝试加载数据和模型 ---")
    try:
        load_data()
        load_model()

        print("\n--- 尝试启动 Flask 服务 ---")
        app.run(host='0.0.0.0', port=5000)

    except Exception as e:
        import traceback

        print("\nFATAL ERROR: API 启动失败！请检查 Traceback。")
        print(f"致命错误信息: {e}")
        traceback.print_exc()
        print("----------------------------")