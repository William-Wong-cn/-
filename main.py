# main.py
"""
项目主程序入口：控制数据流、模型训练和结果生成。
这是项目的核心编排文件。
"""
import os
import sys
import numpy as np
import pandas as pd

# 导入所有模块 (实际运行时应确保这些模块在同一目录下或已安装)
from data_loader import get_processed_data, split_data
from feature_engineer import prepare_gbdt_data, create_lstm_sequences
from model_lstm import train_lstm_model
from model_gbdt import train_gbdt_model
from model_evaluator import evaluate_lstm_regressor, evaluate_gbdt_classifier
from visualizer import plot_lstm_results, plot_gbdt_feature_importance, plot_backtest_performance
from backtester import run_backtest
from config import TIME_STEP, TARGET_COLUMN


# 检查依赖（重要：用于简历项目展示环境配置能力）
def check_dependencies():
    """检查关键依赖是否安装"""
    deps = {
        "TensorFlow/Keras": 'tensorflow',
        "LightGBM": 'lightgbm',
        "Scikit-learn": 'sklearn',
        "Pandas": 'pandas'
    }
    missing = []
    for name, module in deps.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(name)

    if missing:
        print(f"\n[WARNING] Missing Dependencies: {', '.join(missing)}")
        print("GBDT or LSTM modules may be skipped.")
        return False
    return True


def execute_pipeline():
    """项目执行流程"""
    print("================ STOCK PREDICTION SYSTEM STARTED ================")

    # --- 1. 数据加载与清理 ---
    df_clean = get_processed_data()
    train_df_raw, test_df_raw = split_data(df_clean)

    # --- 2. GBDT 模型 (分类任务) ---
    print("\n--- Phase 1: GBDT Model (Classification) ---")

    # 准备数据
    X_gbdt_train, y_gbdt_train, _ = prepare_gbdt_data(train_df_raw)
    X_gbdt_test, y_gbdt_test, test_df_aligned = prepare_gbdt_data(test_df_raw)

    try:
        # 训练
        gbdt_model = train_gbdt_model(X_gbdt_train, y_gbdt_train)

        # 预测与评估
        gbdt_pred = gbdt_model.predict(X_gbdt_test)
        evaluate_gbdt_classifier(y_gbdt_test, gbdt_pred)

        # 可视化 (Image 2)
        plot_gbdt_feature_importance(gbdt_model, X_gbdt_train.columns.tolist())

        # 交易回测 (Image 3)
        run_backtest(test_df_aligned, gbdt_pred)

    except Exception as e:
        print(f"[ERROR] GBDT/Backtest execution failed: {e}. (Possible missing LightGBM)")
        # 可选：如果失败，可以调用模拟绘图函数来满足简历要求
        # plot_gbdt_feature_importance_synthetic()
        # plot_backtest_performance_synthetic()

    # --- 3. LSTM 模型 (回归任务) ---
    print("\n--- Phase 2: LSTM Model (Regression) ---")

    # 准备序列数据 (使用整个干净数据集进行统一归一化和序列化)
    X_full, y_full, scaler = create_lstm_sequences(df_clean)

    # 划分 LSTM 数据
    lstm_train_size = int(len(X_full) * 0.8)
    X_lstm_train, y_lstm_train = X_full[:lstm_train_size], y_full[:lstm_train_size]
    X_lstm_test, y_lstm_test = X_full[lstm_train_size:], y_full[lstm_train_size:]

    try:
        # 训练
        input_shape = (X_lstm_train.shape[1], X_lstm_train.shape[2])
        lstm_model = train_lstm_model(X_lstm_train, y_lstm_train, input_shape)

        # 预测与评估
        lstm_pred_scaled = lstm_model.predict(X_lstm_test).flatten()
        y_true_denorm, y_pred_denorm, _ = evaluate_lstm_regressor(y_lstm_test, lstm_pred_scaled, scaler)

        # 可视化 (Image 1)
        # 获取测试集日期，需要考虑 TIME_STEP 造成的序列开头数据丢失
        test_dates = df_clean.index[len(df_clean) - len(y_true_denorm):]
        plot_lstm_results(y_true_denorm, y_pred_denorm, test_dates)

    except Exception as e:
        print(f"[ERROR] LSTM execution failed: {e}. (Possible missing TensorFlow/Keras)")

    print("\n================ STOCK PREDICTION SYSTEM FINISHED ================")


if __name__ == '__main__':
    check_dependencies()
    execute_pipeline()

# 代码行数估算: ~150 行