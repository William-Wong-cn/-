# backtester.py
"""
交易策略回测模块：基于 GBDT 分类预测结果，模拟交易并评估策略收益。
"""
import pandas as pd
from visualizer import plot_backtest_performance
from config import TARGET_COLUMN


def run_backtest(test_df_aligned, gbdt_predictions):
    """
    运行基于 GBDT 预测的简单多头策略回测。
    策略：预测下一日上涨（1），则持有多头仓位（1）；预测下跌（0），则空仓（0）。
    """
    print("\n--- Backtesting Strategy ---")

    # 确保预测和测试数据对齐
    test_data = test_df_aligned.iloc[len(test_df_aligned) - len(gbdt_predictions):].copy()
    test_data['GBDT_Prediction'] = gbdt_predictions

    # 1. 计算每日收益率
    test_data['Daily_Returns'] = test_data[TARGET_COLUMN].pct_change()

    # 2. 确定策略仓位
    # Position: 1 (持有多头), 0 (空仓)
    # 使用前一天的预测结果 shift(1) 来决定今天的仓位
    test_data['Strategy_Position'] = test_data['GBDT_Prediction'].shift(1).fillna(0)

    # 3. 计算策略每日收益
    # 策略收益 = 市场收益率 * 仓位
    test_data['Strategy_Daily_Returns'] = test_data['Daily_Returns'] * test_data['Strategy_Position']

    # 4. 计算累计收益率
    # 策略累计收益
    test_data['Cumulative_Returns'] = (1 + test_data['Strategy_Daily_Returns'].fillna(0)).cumprod() - 1
    # 基准累计收益 (买入并持有)
    test_data['Buy_Hold_Returns'] = (1 + test_data['Daily_Returns'].fillna(0)).cumprod() - 1

    # 5. 结果分析
    total_strategy_return = test_data['Cumulative_Returns'].iloc[-1]
    total_buy_hold_return = test_data['Buy_Hold_Returns'].iloc[-1]

    print(f"   - Strategy Total Return: {total_strategy_return * 100:.2f}%")
    print(f"   - Buy & Hold Total Return: {total_buy_hold_return * 100:.2f}%")

    # 调用可视化 (Image 3)
    plot_backtest_performance(test_data)

    return test_data

# 代码行数估算: ~85 行