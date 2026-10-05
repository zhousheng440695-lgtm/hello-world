import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json

# ========================== 环境与接口初始化 ==========================
plt.switch_backend('Agg')

TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")
if not TUSHARE_TOKEN:
    raise Exception("环境变量 TUSHARE_TOKEN 不存在，请在GitHub Secrets配置")
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

os.makedirs("data", exist_ok=True)

# ==========================================
# 0. 回测参数设置
# ==========================================
START_YEAR = 2020
END_YEAR = 2025
PORTFOLIO_SIZE = 10  # 每季度选入的股票数量
MOMENTUM_WINDOW = 120  # 动量计算窗口（约 6 个月 / 120 个交易日）

# ==========================================
# 1. 选股核心模块 (截面化极速版)
# ==========================================
def get_quarterly_rebalance_dates(start_year, end_year):
    """获取每年 1、4、7、10 月的第一个交易日作为季度调仓日"""
    cal = pro.trade_cal(exchange='SSE', start_date=f'{start_year}0101', end_date=f'{end_year}1231', is_open='1')
    cal['year_month'] = cal['cal_date'].str[:6]
    # 取每个月的首个交易日
    first_days = cal.groupby('year_month').first().reset_index()
    # 仅保留季初月份
    target_months = ['01', '04', '07', '10']
    quarter_starts = first_days[first_days['year_month'].str[4:6].isin(target_months)]
    return quarter_starts['cal_date'].tolist()

def get_prp_factor_fast(candidate_codes, trade_date):
    """
    终极版 PRP 因子计算：按候选股票池进行分块请求，结合自动重试机制。
    单季度只需 3 次轻量级 API 请求，彻底解决海外 GitHub Actions 网络超时问题。
    """
    print(f"    -> 正在批量拉取 {len(candidate_codes)} 只候选股的 25 个月历史数据计算 PRP...")
    end_dt = pd.to_datetime(trade_date)
    start_dt = end_dt - pd.DateOffset(months=25)
    start_str = start_dt.strftime('%Y%m%d')

    all_monthly_dfs = []
    chunk_size = 50  # 每次请求 50 只股票，完美避开 Tushare 单次请求上限
    
    for i in range(0, len(candidate_codes), chunk_size):
        chunk_codes = ",".join(candidate_codes[i:i + chunk_size])
        
        # 针对跨国网络的 3 次容错重试机制
        retry_count = 3
        while retry_count > 0:
            try:
                # 直接通过 ts_code 列表拉取时间段数据
                df = pro.monthly(ts_code=chunk_codes, start_date=start_str, end_date=trade_date, fields='ts_code,trade_date,pct_chg')
                all_monthly_dfs.append(df)
                break  # 成功则跳出重试循环
            except Exception as e:
                retry_count -= 1
                print(f"       [网络波动] 数据拉取失败，正在重试... 剩余重试次数: {retry_count}")
                time.sleep(2)  # 遇到报错休眠 2 秒再试
        
        time.sleep(0.4)  # 正常的 Tushare 接口频控

    if not all_monthly_dfs:
        return pd.Series(dtype=float)

    all_monthly = pd.concat(all_monthly_dfs, ignore_index=True)

    prp_dict = {}
    for code, group in all_monthly.groupby('ts_code'):
        group = group.sort_values('trade_date').reset_index(drop=True)
        # 至少需要一年的数据才有统计意义
        if len(group) >= 12: 
            prp_dict[code] = group['pct_chg'].autocorr(lag=1)
        else:
            prp_dict[code] = pd.NA

    return pd.Series(prp_dict)

def select_stocks(trade_date):
    """在指定日期进行选股：6个月动量 + 剔除高 PRP"""
    print(f"正在计算 {trade_date} 的选股名单 (季度动量 + PRP风控)...")
    
    # 1. 极速动量计算
    cal = pro.trade_cal(exchange='SSE', start_date='20100101', end_date=trade_date, is_open='1')
    past_date = cal.iloc[-MOMENTUM_WINDOW - 1]['cal_date'] 
    
    df_curr = pro.daily(trade_date=trade_date, fields='ts_code,close')
    time.sleep(0.3)
    df_past = pro.daily(trade_date=past_date, fields='ts_code,close')
    time.sleep(0.3)
    
    df_past.rename(columns={'close': 'past_close'}, inplace=True)
    df = pd.merge(df_curr, df_past, on='ts_code')
    df['momentum'] = (df['close'] / df['past_close']) - 1
    
    status = pro.bak_basic(trade_date=trade_date, fields='ts_code,name,list_status')
    if status.empty:
        status = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
    df = pd.merge(df, status, on='ts_code')
    df = df[~df['name'].str.contains('ST')]
    
    # 2. 动量最强 150 只候选
    candidates = df.sort_values(by='momentum', ascending=False).head(150).copy()
    candidate_codes = candidates['ts_code'].tolist()
    
    # 3. 注入 PRP 因子 (调用全新升级的健壮接口)
    prp_series = get_prp_factor_fast(candidate_codes, trade_date)
    candidates['prp'] = candidates['ts_code'].map(prp_series)
    
    # ================= 恢复严格风控 =================
    # 彻底丢弃那些上市不足 12 个月，或因极端异常拉不到数据的股票
    candidates = candidates.dropna(subset=['prp'])
    # ===============================================
    
    # 4. 剔除 PRP 排名前 30% 的高拥挤度股票
    if not candidates.empty:
        prp_threshold = candidates['prp'].quantile(0.70)
        safe_pool = candidates[candidates['prp'] <= prp_threshold]
    else:
        safe_pool = candidates
        
    # 5. 在安全池中，按动量排序选出最终的 10 只标的
    selected = safe_pool.sort_values(by='momentum', ascending=False).head(PORTFOLIO_SIZE)
    print(f"    -> 截面池: {len(df)} 只 | 动量粗筛: 150 只 | 剔除高PRP后剩余: {len(safe_pool)} 只")
    
    return selected['ts_code'].tolist()

# ==========================================
# 2. 收益计算与评估模块
# ==========================================
def backtest():
    rebalance_dates = get_quarterly_rebalance_dates(START_YEAR, END_YEAR)
    portfolio_daily_returns = pd.Series(dtype=float)
    
    for i in range(len(rebalance_dates) - 1):
        start_date = rebalance_dates[i]
        end_date = rebalance_dates[i + 1]
        
        symbols = select_stocks(start_date)
        print(f"    -> 最终持仓：{symbols}")
        
        prices = pd.DataFrame()
        for ts_code in symbols:
            try:
                df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if not df.empty:
                    df = df.sort_values('trade_date')
                    df.set_index('trade_date', inplace=True)
                    prices[ts_code] = df['close']
            except Exception as e:
                pass
            time.sleep(0.3)
            
        prices.ffill(inplace=True)
        daily_returns = prices.pct_change().fillna(0)
        port_return = daily_returns.mean(axis=1)
        portfolio_daily_returns = pd.concat([portfolio_daily_returns, port_return])
        
    portfolio_daily_returns.index = pd.to_datetime(portfolio_daily_returns.index)
    portfolio_daily_returns.sort_index(inplace=True)
    # 去除首尾重叠的调仓日
    portfolio_daily_returns = portfolio_daily_returns[~portfolio_daily_returns.index.duplicated(keep='first')]
    return portfolio_daily_returns

def calculate_metrics(daily_returns):
    cum_returns = (1 + daily_returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1
    trading_days = len(daily_returns)
    annual_return = (1 + total_return) ** (252 / trading_days) - 1
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    daily_rf = 0.03 / 252
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
    
    # ================= 强制对齐和去重的深证成指基准 (399001.SZ) =================
    szcz = pro.index_daily(ts_code='399001.SZ',
                           start_date=daily_returns.index[0].strftime('%Y%m%d'),
                           end_date=daily_returns.index[-1].strftime('%Y%m%d'))
    szcz.set_index('trade_date', inplace=True)
    szcz.index = pd.to_datetime(szcz.index)
    szcz = szcz.sort_index()
    szcz = szcz[~szcz.index.duplicated(keep='first')]
    
    szcz_returns = szcz['close'].pct_change().fillna(0)
    szcz_cum = (1 + szcz_returns).cumprod()

    print("\n" + "=" * 40)
    print("      量化策略回测报告")
    print("      (季度动量 + PRP防反转)")
    print("=" * 40)
    print(f"测试区间: {daily_returns.index[0].strftime('%Y-%m-%d')} 至 {daily_returns.index[-1].strftime('%Y-%m-%d')}")
    print(f"累计收益率:   {metrics['Total Return'] * 100:.2f}%")
    print(f"年化收益率:   {metrics['Annualized Return'] * 100:.2f}%")
    print(f"最大回撤:     {metrics['Max Drawdown'] * 100:.2f}%")
    print(f"夏普比率:     {metrics['Sharpe Ratio']:.2f}")
    print("=" * 40)

    # ================= 绘图输出 (图例同步修改为 SZSE Component) =================
    plt.figure(figsize=(12, 6))
    plt.plot(cum_returns.index, cum_returns, label='Strategy (Momentum + Low PRP)', color='red')
    plt.plot(szcz_cum.index, szcz_cum, label='SZSE Component Benchmark', color='blue', alpha=0.7)
    plt.title('Strategy Performance vs Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Net Value')
    plt.legend()
    plt.grid(True)
    plt.savefig("data/backtest_plot.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ================= JSON 前端大屏输出 =================
    aligned_data = pd.DataFrame({
        'strategy': cum_returns,
        'benchmark': szcz_cum
    }).ffill().dropna()

    running_max = aligned_data['strategy'].cummax()
    drawdown = (aligned_data['strategy'] - running_max) / running_max * 100

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

    with open("data/backtest_result.json", 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False)

if __name__ == '__main__':
    print("初始化回测引擎...")
    strategy_returns = backtest()
    plot_and_report(strategy_returns)
