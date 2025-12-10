# model_gbdt.py
"""
基于 LightGBM 的梯度提升决策树 (GBDT) 模型模块。
用于股票涨跌的分类预测。
"""
try:
    import lightgbm as lgb
except ImportError:
    print("Warning: LightGBM not installed. GBDT module disabled.")
    lgb = None

import joblib
import os
from config import (GBDT_N_ESTIMATORS, GBDT_MAX_DEPTH, GBDT_LEARNING_RATE,
                    GBDT_RANDOM_STATE, MODEL_SAVE_PATH)

GBDT_MODEL_FILE = os.path.join(MODEL_SAVE_PATH, 'gbdt_model.pkl')


def train_gbdt_model(X_train, y_train):
    """训练 GBDT (LightGBM) 模型并保存 (分类任务)"""
    if lgb is None:
        return None

    print("-> Training GBDT (LightGBM) model...")

    # LightGBM分类器配置
    lgb_clf = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        n_estimators=GBDT_N_ESTIMATORS,
        max_depth=GBDT_MAX_DEPTH,
        learning_rate=GBDT_LEARNING_RATE,
        random_state=GBDT_RANDOM_STATE,
        n_jobs=-1,
        verbose=-1  # 关闭训练过程中的打印
    )

    lgb_clf.fit(X_train, y_train)

    # joblib.dump(lgb_clf, GBDT_MODEL_FILE)
    # print(f"GBDT model saved to {GBDT_MODEL_FILE}")
    return lgb_clf


def load_gbdt_model():
    """加载已保存的 GBDT 模型"""
    if os.path.exists(GBDT_MODEL_FILE) and joblib:
        return joblib.load(GBDT_MODEL_FILE)
    return None

# 代码行数估算: ~60 行