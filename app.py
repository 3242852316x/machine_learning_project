# app.py - 最终修正版 (Streamlit 前端，支持单个和批量预测)

import streamlit as st
import requests
import json
import pandas as pd
import base64
import io

# --- 配置 ---
# 确保后端 Flask API 正在运行在 http://127.0.0.1:5000
API_URL = "http://127.0.0.1:5000/predict"
BULK_API_URL = "http://127.0.0.1:5000/bulk_predict"

# --- 页面布局 ---
st.set_page_config(page_title="天猫复购概率预测", layout="centered")

st.title("🛍️ 天猫复购概率预测 (Streamlit)")
st.markdown("---")

# --- 1. 单个预测部分 ---
st.header("👤 单个预测")

# 输入用户ID
user_id_input = st.number_input(
    "用户ID (user_id):",
    min_value=1,
    value=328862,  # 示例值
    step=1
)

# 输入商家ID
merchant_id_input = st.number_input(
    "商家ID (merchant_id):",
    min_value=1,
    value=2882,  # 示例值
    step=1
)

if st.button("预测复购概率"):

    payload = {
        "user_id": user_id_input,
        "merchant_id": merchant_id_input
    }

    st.info("正在发送预测请求到后端 API...")

    try:
        response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            prob = result.get('repurhase_prob')

            st.success("✅ 预测成功！")

            st.metric(
                label=f"用户 {user_id_input} 对商家 {merchant_id_input} 的复购概率",
                value=f"{prob * 100:.2f}%"
            )

            if prob > 0.5:
                st.balloons()
                st.write("🎉 **高概率复购！** 建议进行营销活动以提高转化率。")
            elif prob > 0.2:
                st.write("📈 **中等复购潜力。** 建议谨慎投入营销资源。")
            else:
                st.write("📉 **低概率复购。** 建议降低营销优先级。")

        else:
            error_data = response.json()
            st.error(f"❌ API 预测失败 (状态码: {response.status_code})")
            st.json(error_data)

    except requests.exceptions.ConnectionError:
        st.error(f"❌ 连接错误：无法连接到后端 API ({API_URL})。请确保您的 `api.py` 正在运行。")
    except Exception as e:
        st.error(f"发生未知错误: {e}")

# --- 2. 批量预测部分 ---
st.markdown("---")
st.header("📂 批量预测 (上传 CSV)")
st.caption("上传包含 'user_id' 和 'merchant_id' 两列的 CSV 文件进行批量预测。")

uploaded_file = st.file_uploader("选择 CSV 文件", type="csv")

if uploaded_file is not None:
    # 读取文件内容，准备发送给后端
    csv_bytes = uploaded_file.getvalue()
    csv_string = csv_bytes.decode('utf-8')

    if st.button("开始批量预测"):
        st.info(f"正在上传并预测 {uploaded_file.name} 中的数据...")

        try:
            # 发送 POST 请求到 /bulk_predict 路由，发送 CSV 文本
            response = requests.post(BULK_API_URL, data=csv_string, timeout=60)  # 批量预测增加超时时间

            if response.status_code == 200:
                results_list = response.json()
                st.success(f"✅ 批量预测成功！共处理 {len(results_list)} 条记录。")

                # 将结果转为 DataFrame 并展示
                results_df = pd.DataFrame(results_list)
                results_df.rename(columns={'prob': '复购概率'}, inplace=True)

                st.subheader("预测结果概览")
                st.dataframe(results_df)


                # 提供下载链接
                @st.cache_data
                def convert_df(df):
                    return df.to_csv(index=False).encode('utf-8')


                csv_download = convert_df(results_df)

                st.download_button(
                    label="📥 下载预测结果 CSV",
                    data=csv_download,
                    file_name='bulk_prediction_results.csv',
                    mime='text/csv',
                )

            else:
                error_data = response.json()
                st.error(f"❌ 批量预测失败 (状态码: {response.status_code})")
                st.write("后端返回错误详情：")
                st.json(error_data)

        except requests.exceptions.ConnectionError:
            st.error(f"❌ 连接错误：无法连接到后端 API ({BULK_API_URL})。请确保您的 `api.py` 正在运行。")
        except Exception as e:
            st.error(f"发生未知错误: {e}")