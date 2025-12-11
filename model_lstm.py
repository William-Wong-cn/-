# encoding=utf-8
"""
基于 Keras/TensorFlow 的 LSTM 神经网络模型模块。
用于股票价格的连续值预测 (回归)。
"""
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except ImportError:
    print("Warning: TensorFlow/Keras not installed. LSTM module disabled.")
    # 定义占位符以避免 NameError
    Sequential, load_model, LSTM, Dense, Dropout = None, None, None, None, None

import os
from config import (LSTM_UNITS, LSTM_DROPOUT, LSTM_EPOCHS, LSTM_BATCH_SIZE,
                    MODEL_SAVE_PATH)

LSTM_MODEL_FILE = os.path.join(MODEL_SAVE_PATH, 'lstm_model.h5')


def build_lstm_model(input_shape):
    """
    构建双层 LSTM 神经网络模型。
    input_shape: (time_step, feature_count)
    """
    if not Sequential:
        return None

    print("-> Building LSTM model...")
    model = Sequential([
        # 1. 第一层 LSTM：返回序列
        LSTM(units=LSTM_UNITS, return_sequences=True, input_shape=input_shape, name='LSTM_1'),
        Dropout(LSTM_DROPOUT, name='Dropout_1'),

        # 2. 第二层 LSTM：不返回序列 (输出给全连接层)
        LSTM(units=LSTM_UNITS, name='LSTM_2'),
        Dropout(LSTM_DROPOUT, name='Dropout_2'),

        # 3. 全连接层 (输出单个预测值)
        Dense(units=1, activation='linear', name='Output')
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    # model.summary() # 在实际项目中应打印，此处省略以减少控制台输出

    return model


def train_lstm_model(X_train, y_train, input_shape):
    """训练 LSTM 模型并保存"""
    model = build_lstm_model(input_shape)
    if model is None:
        print("Error: LSTM model cannot be built.")
        return None

    print(f"-> Training LSTM model for {LSTM_EPOCHS} epochs...")
    model.fit(X_train, y_train,
              epochs=LSTM_EPOCHS,
              batch_size=LSTM_BATCH_SIZE,
              verbose=1,
              shuffle=True)  # 打乱训练数据

    # 模拟保存
    # model.save(LSTM_MODEL_FILE)
    # print(f"LSTM model saved to {LSTM_MODEL_FILE}")
    return model


def load_lstm_model():
    """加载已保存的 LSTM 模型"""
    if os.path.exists(LSTM_MODEL_FILE) and load_model:
        return load_model(LSTM_MODEL_FILE)
    return None

