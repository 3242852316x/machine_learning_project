# 🛍️ 天猫复购预测系统 (Tmall Repurchase Prediction)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LightGBM-2.3.1-orange.svg" alt="LightGBM">
  <img src="https://img.shields.io/badge/FastAPI-Latest-009688.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Pandas-0.23+-green.svg" alt="Pandas">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

## 📖 项目简介
本项目是一个基于天猫用户行为日志的**全栈机器学习应用**。通过挖掘用户的历史交互数据（点击、购买、加购等），利用 **LightGBM** 算法精准预测用户在未来产生“复购”行为的概率。

项目不仅包含完整的模型训练 Pipeline，还提供了一个基于 **FastAPI** 的生产级预测接口及 Web 测试界面。

---

## 🛠️ 核心技术栈
* **模型算法**: `LightGBM` (基于梯度提升决策树的高效分类模型)。
* **后端框架**: `FastAPI` (高性能异步 Python Web 框架)。
* **数据处理**: `Pandas` (针对旧版本进行了 `.agg` 兼容性语法优化)。
* **模型部署**: `Joblib` (模型序列化与持久化)。
* **前端展示**: 原生 `HTML5` + `JavaScript` (Fetch API 实时交互)。

---

## 🔄 项目全流程架构

```mermaid
graph TD
    A[原始日志数据] --> B{特征工程}
    B -->|构建 11 维核心特征| C[LightGBM 训练]
    C --> D[保存模型 .pkl]
    D --> E[FastAPI 后端服务]
    E --> F[Web 交互前端]
    F -->|POST 请求| E
    E -->|预测概率| F
