# encoding=utf-8
"""
数据加载、初步清洗和训练/测试集划分模块。
核心功能：处理 NaN 值和格式转换。
"""
import pandas as pd
from config import DATA_FILE_PATH, TRAIN_RATIO


def load_raw_data(file_path):
    """从 CSV 文件加载原始数据"""
    try:
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}.")
        return None

    # 确保索引是日期时间格式
    df.index = pd.to_datetime(df.index)

    print(f"Raw data loaded. Shape: {df.shape}")
    return df


def clean_data(df):
    """执行关键数据清洗和缺失值处理"""
    if df is None:
        return None

    print("-> Data Cleaning: Handling NaN values...")

    # 1. 关键列 NaN 处理（如 Close, Volume - 随机插入的 NaN）
    critical_cols = ['Close', 'Volume']
    initial_nan_rows = df[critical_cols].isnull().any(axis=1).sum()
    df.dropna(subset=critical_cols, inplace=True)
    print(f"   - Dropped {initial_nan_rows} rows due to NaN in critical columns (Close/Volume).")

    # 2. 技术指标列 NaN 处理（如 SMA_50, MACD 等 - 天然产生的 NaN）
    final_nan_rows = df.isnull().any(axis=1).sum()
    df.dropna(inplace=True)
    print(f"   - Dropped remaining {final_nan_rows} rows (mostly initial indicator NaNs).")

    # 3. 格式优化
    df['Volume'] = df['Volume'].astype(float)

    print(f"Clean data shape: {df.shape}")
    return df


def split_data(df):
    """根据时间点进行训练集和测试集划分"""
    train_size = int(len(df) * TRAIN_RATIO)
    train_data = df[:train_size]
    test_data = df[train_size:]

    print(f"Data split ratio: Train {TRAIN_RATIO}, Test {1 - TRAIN_RATIO}")
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    return train_data, test_data


def get_processed_data():
    """主接口函数"""
    df_raw = load_raw_data(DATA_FILE_PATH)
    df_clean = clean_data(df_raw)

    if df_clean is None or df_clean.empty:
        raise ValueError("Data processing resulted in an empty DataFrame.")

    return df_clean
