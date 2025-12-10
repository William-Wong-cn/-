# visualizer.py
"""
结果可视化模块：生成所有项目所需的图表（至少三张）。
"""
import matplotlib.pyplot as plt
import pandas as pd
import os
from config import RESULT_SAVE_PATH, IMAGE_1_NAME, IMAGE_2_NAME, IMAGE_3_NAME


def save_plot(fig, filename):
    """保存图表到结果路径"""
    save_path = os.path.join(RESULT_SAVE_PATH, filename)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"   - Image saved to {save_path}")


def plot_lstm_results(y_true, y_pred, dates):
    """生成图像 1：LSTM 预测与真实价格对比图 (回归)"""
    print("-> Generating Image 1: LSTM Prediction vs Actual...")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(dates, y_true, label='Actual Price', color='blue', linewidth=2)
    ax.plot(dates, y_pred, label='LSTM Predicted Price', color='red', linestyle='--', linewidth=1.5)

    ax.set_title('LSTM Model: Actual vs Predicted Stock Price (Regression)', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Stock Price', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    save_plot(fig, IMAGE_1_NAME)


def plot_gbdt_feature_importance(model, feature_names):
    """生成图像 2：GBDT 特征重要性图 (分类)"""
    print("-> Generating Image 2: GBDT Feature Importance...")
    try:
        importances = model.feature_importances_
        feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_df = feature_df.sort_values(by='Importance', ascending=False).head(10)

        if feature_df['Importance'].sum() == 0:
            print("Warning: Feature importance sum is zero. Skipping plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_df['Feature'], feature_df['Importance'], color='skyblue')
        ax.set_xlabel('Feature Importance Score', fontsize=12)
        ax.set_title('Top 10 GBDT Feature Importance for Stock Direction', fontsize=16)
        ax.invert_yaxis()
        fig.tight_layout()
        save_plot(fig, IMAGE_2_NAME)

    except AttributeError:
        print("Warning: GBDT model does not support feature_importances_. Skipping plot.")


def plot_backtest_performance(history_df, strategy_name="GBDT_Classifier_Strategy"):
    """生成图像 3：回测性能曲线图"""
    print("-> Generating Image 3: Backtest Performance Curve...")

    history_df['Cumulative_Returns_Pct'] = history_df['Cumulative_Returns'] * 100
    history_df['Buy_Hold_Returns_Pct'] = history_df['Buy_Hold_Returns'] * 100

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(history_df.index, history_df['Cumulative_Returns_Pct'],
            label=f'{strategy_name} Returns', color='green', linewidth=2)
    ax.plot(history_df.index, history_df['Buy_Hold_Returns_Pct'],
            label='Buy & Hold Benchmark', color='gray', linestyle='--', linewidth=1.5)

    ax.set_title(f'Trading Strategy Backtest Performance (Strategy vs Benchmark)', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Returns (%)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    save_plot(fig, IMAGE_3_NAME)

# 代码行数估算: ~120 行