# encoding=utf-8
"""
特征工程、数据标准化和时间序列转换模块。
核心功能：为 GBDT 生成分类目标；为 LSTM 创建序列数据。
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config import TARGET_COLUMN, TIME_STEP, GLOBAL_SCALER_STORE


def prepare_gbdt_data(data):
    """
    为 GBDT 模型准备数据 (分类任务)。
    特征：所有技术指标和 OHLCV。
    目标：未来一天的涨跌（1: 涨, 0: 跌/平）。
    """
    print("-> Feature Engineering for GBDT: Generating Classification Target...")
    df = data.copy()

    # 目标变量：未来1日收盘价的涨跌 (分类)
    # shift(-1) 表示下一行的数据 (即未来的价格)
    df['Target_Class'] = (df[TARGET_COLUMN].shift(-1) > df[TARGET_COLUMN]).astype(int)

    # 清理掉 Target_Class 产生的最后一个 NaN 值
    df.dropna(subset=['Target_Class'], inplace=True)

    # 特征选择：排除日期索引，和目标强耦合的列
    features_to_drop = [
        'Target_Class',
        TARGET_COLUMN,
        # 移除 Daily_Return, 因为它和目标 Target_Class 强相关，可能导致数据泄露。
        'Daily_Return',
    ]

    X = df.drop(columns=[col for col in features_to_drop if col in df.columns])
    y = df['Target_Class']

    print(f"   - GBDT Feature columns: {X.columns.tolist()}")
    print(f"   - GBDT Data shape: X:{X.shape}, Y:{y.shape}")

    return X, y, df


def create_lstm_sequences(data, time_step=TIME_STEP):
    """
    为 LSTM 模型创建序列数据 (回归任务)。
    对所有特征进行归一化，并切割成 [样本数, 时间步长, 特征数] 的三维张量。
    """
    print(f"-> Feature Engineering for LSTM: Creating Sequences (Time Step={time_step})...")

    features = data.columns.tolist()
    df_features = data[features]

    # 归一化 (Scaler只在首次Fit，确保一致性)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df_features)
    GLOBAL_SCALER_STORE['main'] = scaler
    GLOBAL_SCALER_STORE['features'] = features

    X, y = [], []
    target_index = features.index(TARGET_COLUMN)  # 找到目标列的索引

    for i in range(time_step, len(df_scaled)):
        # X: time_step 个历史数据序列
        X.append(df_scaled[i - time_step:i, :])
        # Y: 序列结束时的下一个收盘价 (下一个时间步的目标)
        y.append(df_scaled[i, target_index])

    X = np.array(X)
    y = np.array(y)

    print(f"   - LSTM 3D Tensor shape: X:{X.shape}, Y:{y.shape}")
    return X, y, scaler
