import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ========================== 修复：matplotlib 无界面后端（GitHub Linux环境必须） ==========================
plt.switch_backend('Agg')

# ========================== 修复：从环境变量读取token，不要硬编码！ ==========================
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")
if not TUSHARE_TOKEN:
    raise Exception("环境变量 TUSHARE_TOKEN 不存在，请在GitHub Secrets配置")
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# 自动创建data输出目录
os.makedirs("data", exist_ok=True)

# ==========================================
# 0. 初始化与基础设置
# ==========================================
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
    """在指定日期进行选股：高ROE + 高股息"""
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
    # 先把无股息的排除，并按股息率从高到低排，只取前150候选
    df = df[df['dv_ratio'] > 0].sort_values(by='dv_ratio', ascending=False).head(150)
    # 2. 获取上一年的年报财务指标（ROE）
    report_period = f"{int(trade_date[:4]) - 1}1231"
    target_codes = df['ts_code'].tolist()
    fina_list = []
    chunk_size = 50
    for i in range(0, len(target_codes), chunk_size):
        chunk_codes = ",".join(target_codes[i:i + chunk_size])
        try:
            temp_fina = pro.fina_indicator(ts_code=chunk_codes, period=report_period, fields='ts_code,roe')
            fina_list.append(temp_fina)
        except Exception as e:
            print(f"拉取财务数据时出错: {e}")
        time.sleep(0.3)
    fina = pd.concat(fina_list, ignore_index=True)
    fina = fina.drop_duplicates(subset=['ts_code'])
    # 合并数据，ROE>15%取前10
    pool = pd.merge(df, fina, on='ts_code')
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
        symbols = select_stocks(start_date)
        print(f"本次选中股票：{symbols}")
        prices = pd.DataFrame()
        for ts_code in symbols:
            try:
                df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if not df.empty:
                    df = df.sort_values('trade_date')
                    df.set_index('trade_date', inplace=True)
                    prices[ts_code] = df['close']
            except Exception as e:
                print(f"股票 {ts_code} 行情拉取失败: {e}")
            time.sleep(0.3)
        prices.ffill(inplace=True)
        daily_returns = prices.pct_change().fillna(0)
        port_return = daily_returns.mean(axis=1)
        portfolio_daily_returns = pd.concat([portfolio_daily_returns, port_return])
    portfolio_daily_returns.index = pd.to_datetime(portfolio_daily_returns.index)
    portfolio_daily_returns.sort_index(inplace=True)
    portfolio_daily_returns = portfolio_daily_returns[~portfolio_daily_returns.index.duplicated(keep='first')]
    return portfolio_daily_returns

# ==========================================
# 3. 绩效评价与报告输出
# ==========================================
def calculate_metrics(daily_returns):
    cum_returns = (1 + daily_returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1
    trading_days = len(daily_returns)
    annual_return = (1 + total_return) ** (252 / trading_days) - 1
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    risk_free_rate = 0.03
    daily_rf = risk_free_rate / 252
    excess_returns = daily_returns - daily_rf
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    return {
        "Total Return": total_return,
        "Annualized Return": annual_return,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe_ratio
    }, cum_returns

def plot_and_report(daily_returns):
    metrics, cum_returns = calculate_metrics(daily_returns)
    
    # ================= 1. 获取沪深300基准并彻底去重 =================
    hs300 = pro.index_daily(ts_code='000300.SH',
                            start_date=daily_returns.index[0].strftime('%Y%m%d'),
                            end_date=daily_returns.index[-1].strftime('%Y%m%d'))
    hs300.set_index('trade_date', inplace=True)
    hs300.index = pd.to_datetime(hs300.index)
    hs300 = hs300.sort_index()
    
    # 核心修复：强行剔除 Tushare 可能返回的重复日期，防止 pd.DataFrame 崩溃
    hs300 = hs300[~hs300.index.duplicated(keep='first')]
    
    hs300_returns = hs300['close'].pct_change().fillna(0)
    hs300_cum = (1 + hs300_returns).cumprod()

    # 打印终端报告
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

    # 绘图保存图片文件
    plt.figure(figsize=(12, 6))
    plt.plot(cum_returns.index, cum_returns, label='Strategy (High ROE + Div)', color='red')
    plt.plot(hs300_cum.index, hs300_cum, label='HS300 Benchmark', color='blue', alpha=0.7)
    plt.title('Strategy Performance vs Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Net Value')
    plt.legend()
    plt.grid(True)
    plt.savefig("data/backtest_plot.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ================= 2. 输出 Streamlit 适用的嵌套 JSON =================
    # 两边都已去重，此时 pd.DataFrame 绝对不会再报错
    aligned_data = pd.DataFrame({
        'strategy': cum_returns,
        'benchmark': hs300_cum
    }).ffill().dropna()

    # 计算用于前端水下曲线展示的动态回撤
    running_max = aligned_data['strategy'].cummax()
    drawdown = (aligned_data['strategy'] - running_max) / running_max * 100

    # 构建包含 metrics 键的极简数据结构
    output = {
        "metrics": {
            "total_return": round(metrics['Total Return'] * 100, 2),
            "annual_return": round(metrics['Annualized Return'] * 100, 2),
            "max_drawdown": round(metrics['Max Drawdown'] * 100, 2),
            "sharpe_ratio": round(metrics['Sharpe Ratio'], 2)
        },
        "chart_data": {
            "dates": aligned_data.index.strftime('%Y-%m-%d').tolist(),
            "strategy": aligned_data['strategy'].round(4).tolist(),
            "benchmark": aligned_data['benchmark'].round(4).tolist(),
            "drawdown": drawdown.round(2).tolist()
        }
    }

    # 写入单一 JSON 文件
    with open("data/backtest_result.json", 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False)

    print("✅ 回测结果已保存至 data/backtest_result.json")
    print("✅ 净值曲线图片保存至 data/backtest_plot.png")

if __name__ == '__main__':
    print("初始化回测引擎，拉取财务数据...")
    strategy_returns = backtest()
    plot_and_report(strategy_returns)
