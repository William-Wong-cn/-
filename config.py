# config.py
"""
项目配置模块：定义路径、超参数和常量。
"""
import os

# --- 路径配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
DATA_FILE_NAME = 'stock_data.csv'
DATA_FILE_PATH = os.path.join(BASE_DIR, DATA_FILE_NAME)
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models')
RESULT_SAVE_PATH = os.path.join(BASE_DIR, 'results')

# 创建必要的文件夹
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULT_SAVE_PATH, exist_ok=True)

# --- 数据配置 ---
TARGET_COLUMN = 'Close'  # 预测目标列
TIME_STEP = 60           # LSTM时间步长 (用前60天数据预测下一天)
TRAIN_RATIO = 0.8        # 训练集比例

# --- 模型参数 ---
# LSTM (神经网络)
LSTM_UNITS = 64
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 10         # 实际项目中应更高，此处为演示保持较低
LSTM_BATCH_SIZE = 32

# GBDT (机器学习) - 使用LightGBM/XGBoost
GBDT_N_ESTIMATORS = 200
GBDT_MAX_DEPTH = 6
GBDT_LEARNING_RATE = 0.05
GBDT_RANDOM_STATE = 42

# --- 可视化配置 ---
IMAGE_1_NAME = '1_LSTM_Prediction_vs_Actual.png'
IMAGE_2_NAME = '2_GBDT_Feature_Importance.png'
IMAGE_3_NAME = '3_Backtest_Performance.png'

# --- 全局变量（用于Scaler共享） ---
# 这是一个字典，用于在 feature_engineer 中存储 MinMaxScaer 实例
GLOBAL_SCALER_STORE = {}

def get_config():
    """返回配置字典"""
    return {
        'TARGET_COLUMN': TARGET_COLUMN,
        'TIME_STEP': TIME_STEP,
        'TRAIN_RATIO': TRAIN_RATIO,
        'LSTM_UNITS': LSTM_UNITS,
        'LSTM_EPOCHS': LSTM_EPOCHS,
        'GBDT_N_ESTIMATORS': GBDT_N_ESTIMATORS,
        'DATA_FILE_PATH': DATA_FILE_PATH
    }

print("Config loaded successfully.")

# 代码行数估算: ~50 行