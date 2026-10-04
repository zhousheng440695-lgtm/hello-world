import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# 0. 初始化与基础设置
# ==========================================
ts.set_token('TUSHARE_TOKEN')
pro = ts.pro_api()

# 设置回测参数
START_YEAR = 2020
END_YEAR = 2025
PORTFOLIO_SIZE = 10  # 每年选入的股票数量
INITIAL_CAPITAL = 1000000


# ==========================================
# 1. 核心数据获取与选股模块
# ==========================================
def get_rebalance_dates(start_year, end_year):
    """获取每年 5 月的第一个交易日作为调仓日"""
    dates = []
    for year in range(start_year, end_year + 1):
        cal = pro.trade_cal(exchange='SSE', start_date=f'{year}0501', end_date=f'{year}0515', is_open='1')
        dates.append(cal.iloc[0]['cal_date'])
    return dates


def select_stocks(trade_date):
    """在指定日期进行选股：高ROE + 高股息 (经过 Tushare 接口优化)"""
    print(f"正在计算 {trade_date} 的选股名单...")

    # 1. 获取当日的基本面数据（包含股息率 dv_ratio）
    daily_basic = pro.daily_basic(trade_date=trade_date, fields='ts_code,close,dv_ratio')

    # 获取当日股票状态（过滤ST和停牌）
    status = pro.bak_basic(trade_date=trade_date, fields='ts_code,name,list_status')
    if status.empty:
        status = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')

    # 合并基本信息，过滤 ST 股
    df = pd.merge(daily_basic, status, on='ts_code')
    df = df[~df['name'].str.contains('ST')]

    # ================= 核心优化逻辑 =================
    # 先把无股息的排除，并按股息率从高到低排，只取前 150-200 名候选者。
    # 因为我们最终只需要 10 只股票，前 150 名高股息里绝对能找出 10 只 ROE > 15% 的。
    # 这样极大减少了接下来获取财务指标的 API 请求压力。
    df = df[df['dv_ratio'] > 0].sort_values(by='dv_ratio', ascending=False).head(150)

    # 2. 获取上一年的年报财务指标（ROE）
    report_period = f"{int(trade_date[:4]) - 1}1231"

    # 将候选的 150 只股票代码分批转为逗号分隔的字符串
    target_codes = df['ts_code'].tolist()
    fina_list = []
    chunk_size = 50  # Tushare 允许多个 code 一起查，每次传入 50 个

    for i in range(0, len(target_codes), chunk_size):
        chunk_codes = ",".join(target_codes[i:i + chunk_size])

        # 此时传入了必填参数 ts_code
        try:
            temp_fina = pro.fina_indicator(ts_code=chunk_codes, period=report_period, fields='ts_code,roe')
            fina_list.append(temp_fina)
        except Exception as e:
            print(f"拉取财务数据时出错: {e}")

        time.sleep(0.3)  # 停顿 0.3 秒，遵守 Tushare 的频控规则

    fina = pd.concat(fina_list, ignore_index=True)
    fina = fina.drop_duplicates(subset=['ts_code'])

    # 3. 合并数据并执行最后的策略逻辑
    pool = pd.merge(df, fina, on='ts_code')

    # 此时池子里已经按股息率排好序了，只需要过滤 ROE > 15%，然后切片取前 10 只即可
    selected = pool[pool['roe'] > 15].head(PORTFOLIO_SIZE)

    return selected['ts_code'].tolist()


# ==========================================
# 2. 收益计算模块
# ==========================================
def backtest():
    rebalance_dates = get_rebalance_dates(START_YEAR, END_YEAR)
    portfolio_daily_returns = pd.Series(dtype=float)

    for i in range(len(rebalance_dates) - 1):
        start_date = rebalance_dates[i]
        end_date = rebalance_dates[i + 1]

        # 1. 获取当期选股名单
        symbols = select_stocks(start_date)

        # 2. 获取这些股票在持有期内的日线行情（计算区间收益）
        prices = pd.DataFrame()
        for ts_code in symbols:
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if not df.empty:
                df = df.sort_values('trade_date')
                df.set_index('trade_date', inplace=True)
                prices[ts_code] = df['close']
            time.sleep(0.3)  # 遵守 Tushare 接口频控

        # 3. 处理停牌缺失值，计算每日收益率
        prices.ffill(inplace=True)
        daily_returns = prices.pct_change().fillna(0)

        # 4. 等权重组合：组合日收益率为各股收益率的平均值
        port_return = daily_returns.mean(axis=1)
        portfolio_daily_returns = pd.concat([portfolio_daily_returns, port_return])

    # 按日期排序
    portfolio_daily_returns.index = pd.to_datetime(portfolio_daily_returns.index)
    portfolio_daily_returns.sort_index(inplace=True)
    return portfolio_daily_returns


# ==========================================
# 3. 绩效评价与报告输出
# ==========================================
def calculate_metrics(daily_returns):
    """计算夏普比率、最大回撤等指标"""
    # 累计净值
    cum_returns = (1 + daily_returns).cumprod()

    # 1. 年化收益率 (假设每年252个交易日)
    total_return = cum_returns.iloc[-1] - 1
    trading_days = len(daily_returns)
    annual_return = (1 + total_return) ** (252 / trading_days) - 1

    # 2. 最大回撤
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    # 3. 夏普比率 (无风险利率设为 3%)
    risk_free_rate = 0.03
    daily_rf = risk_free_rate / 252
    excess_returns = daily_returns - daily_rf
    # 年化夏普 = 日均超额收益 / 日超额收益标准差 * sqrt(252)
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

    return {
        "Total Return": total_return,
        "Annualized Return": annual_return,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe_ratio
    }, cum_returns


def plot_and_report(daily_returns):
    metrics, cum_returns = calculate_metrics(daily_returns)

    # 获取同期沪深300作为基准
    hs300 = pro.index_daily(ts_code='000300.SH', start_date=daily_returns.index[0].strftime('%Y%m%d'),
                            end_date=daily_returns.index[-1].strftime('%Y%m%d'))
    hs300.set_index('trade_date', inplace=True)
    hs300.index = pd.to_datetime(hs300.index)
    hs300 = hs300.sort_index()
    hs300_returns = hs300['close'].pct_change().fillna(0)
    hs300_cum = (1 + hs300_returns).cumprod()

    # 打印回测报告
    print("\n" + "=" * 40)
    print("      基本面量化策略回测报告")
    print("      (高 ROE + 高股息轮动)")
    print("=" * 40)
    print(f"测试区间: {daily_returns.index[0].strftime('%Y-%m-%d')} 至 {daily_returns.index[-1].strftime('%Y-%m-%d')}")
    print(f"累计收益率:   {metrics['Total Return'] * 100:.2f}%")
    print(f"年化收益率:   {metrics['Annualized Return'] * 100:.2f}%")
    print(f"最大回撤:     {metrics['Max Drawdown'] * 100:.2f}%")
    print(f"夏普比率:     {metrics['Sharpe Ratio']:.2f}")
    print("=" * 40)

    # 绘制收益曲线
    plt.figure(figsize=(12, 6))
    plt.plot(cum_returns.index, cum_returns, label='Strategy (High ROE + Div)', color='red')
    plt.plot(hs300_cum.index, hs300_cum, label='HS300 Benchmark', color='blue', alpha=0.7)
    plt.title('Strategy Performance vs Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Net Value')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # 运行回测并输出报告
    print("初始化回测引擎，拉取财务数据...")
    strategy_returns = backtest()
    plot_and_report(strategy_returns)
