# encoding=utf-8
"""
模型评估模块：计算回归模型的 RMSE/MAE 和分类模型的准确率等指标。
"""
import numpy as np
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, \
    classification_report
from config import TARGET_COLUMN, GLOBAL_SCALER_STORE


def inverse_transform_data(y_scaled, scaler, target_column_name=TARGET_COLUMN):
    """
    将归一化后的数据反转回原始价格尺度。
    适用于 LSTM 的预测结果。
    """
    target_index = GLOBAL_SCALER_STORE['features'].index(target_column_name)
    num_features = scaler.scale_.size

    # 构造一个包含所有特征的虚拟数组
    dummy_array = np.zeros((len(y_scaled), num_features))
    dummy_array[:, target_index] = y_scaled.flatten()

    # 反归一化并提取目标列
    y_denorm = scaler.inverse_transform(dummy_array)[:, target_index]
    return y_denorm


def evaluate_lstm_regressor(y_true_scaled, y_pred_scaled, scaler):
    """评估 LSTM 回归模型"""
    print("\n--- Model Evaluation: LSTM Regression ---")

    # 反归一化结果
    y_true_denorm = inverse_transform_data(y_true_scaled, scaler)
    y_pred_denorm = inverse_transform_data(y_pred_scaled, scaler)

    # 计算指标
    rmse = math.sqrt(mean_squared_error(y_true_denorm, y_pred_denorm))
    mae = mean_absolute_error(y_true_denorm, y_pred_denorm)

    print(f"   - RMSE (Root Mean Squared Error): ${rmse:,.4f}")
    print(f"   - MAE (Mean Absolute Error): ${mae:,.4f}")

    return y_true_denorm, y_pred_denorm, {'RMSE': rmse, 'MAE': mae}


def evaluate_gbdt_classifier(y_true, y_pred):
    """评估 GBDT 分类模型"""
    print("\n--- Model Evaluation: GBDT Classification ---")

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    # 关键指标 (用于展示专业性)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"   - Accuracy: {accuracy:.4f}")
    print(f"   - Precision (上涨预测准确率): {precision:.4f}")
    print(f"   - Recall (上涨捕获率): {recall:.4f}")
    print(f"   - F1 Score: {f1:.4f}")
    print("   - Confusion Matrix:\n", cm)

    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}

