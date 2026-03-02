# 🛍️ 天猫复购预测系统 (Tmall Repurchase Prediction)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LightGBM-2.3.1-orange.svg" alt="LightGBM">
  <img src="https://img.shields.io/badge/FastAPI-Latest-009688.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Pandas-0.23+-green.svg" alt="Pandas">
</p>

## 📖 项目概述
本项目是一个端到端的人工智能应用，旨在通过分析天猫用户的历史行为日志，预测其在未来 6 个月内是否会再次购买特定商家的商品。项目完成了从**原始数据清洗、高性能特征工程、模型训练优化**到** RESTful API 部署**的全流程。

---

## 🧠 核心逻辑详解

### 1. 数据预处理与加载
系统首先加载海量的用户行为日志（包括 `user_id`, `item_id`, `cat_id`, `seller_id`, `action_type`, `time_stamp`）。

### 2. 特征工程优化 (V3 核心逻辑)
这是项目的灵魂。为了解决 Pandas 版本差异导致的聚合错误（如 `aggregate() missing 1 required positional argument`），我们采用了**显式字典聚合语法**：



* **交互频率 (Frequency)**：统计用户与商家的总交互次数、点击数、购买数。
* **近因特征 (Recency)**：计算 `last_active_days`（距离双十一最近的一次交互时间）。该值越小，复购概率通常越高。
* **转化指标 (Conversion)**：计算 `buy_rate`（购买次数 / 总行为次数），量化用户的购买果断程度。
* **多样性特征**: 统计用户购买过的唯一商品数 (`unique_item_count`) 和唯一品类数 (`unique_cat_count`)。

### 3. 模型训练 (LightGBM)
采用 **LightGBM** 算法处理非平衡数据集（本项目正样本比例约 6.11%）：
* **训练规模**: 约 260,864 条样本。
* **特征维度**: 自动筛选出 10-11 个最强预测能力的特征。
* **兼容性处理**: 特别适配了 Scikit-learn 0.22.2 版本，解决了 `_num_features` 等 API 缺失导致的训练中断问题。

### 4. 全栈预测部署
* **后端 (FastAPI)**: 封装模型，通过 `POST /predict` 接口接收 JSON 数据，支持跨域请求 (CORS)。
* **前端 (Web UI)**: 提供基于 Vanilla JS 的交互界面，用户输入特征后可实时获取复购概率百分比。

---

## 🛠️ 技术栈
* **数据科学**: `Pandas`, `NumPy`, `Scikit-learn`
* **机器学习**: `LightGBM`
* **API 开发**: `FastAPI`, `Uvicorn`, `Pydantic`
* **模型固化**: `Joblib`
* **前端**: `HTML5`, `CSS3`, `JavaScript` (Fetch API)
