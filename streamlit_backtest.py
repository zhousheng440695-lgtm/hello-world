import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go

# ==========================================
# 1. 页面基础配置
# ==========================================
st.set_page_config(
    page_title="量化策略监控面板",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("📈 量化策略监控塔")
# ========== 核心修改 1：更新页面副标题 ==========
st.markdown("基于 `Tushare` 数据源的截面轮动策略 **(季度动量 + PRP防反转)**")

# ==========================================
# 2. 数据读取与解析
# ==========================================
# ========== 核心修改 2：加入 ttl=0 强制每次刷新都读取最新 JSON，防止缓存卡死 ==========
@st.cache_data(ttl=0)
def load_data(file_path="data/backtest_result.json"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"未找到数据文件 {file_path}，请确保回测 Python 脚本已运行并生成了该 JSON 文件。")
        st.stop()

data = load_data()
metrics = data['metrics']
chart_data = data['chart_data']

# ==========================================
# 3. 核心指标卡片 (Metrics)
# ==========================================
st.markdown("### 📊 核心绩效指标")
col1, col2, col3, col4 = st.columns(4)

def format_color(val):
    return "normal" if float(val) >= 0 else "inverse"

col1.metric("累计收益率", f"{metrics['total_return']}%", 
            delta=f"{metrics['total_return']}%", delta_color=format_color(metrics['total_return']))
col2.metric("年化收益率", f"{metrics['annual_return']}%", 
            delta=f"{metrics['annual_return']}%", delta_color=format_color(metrics['annual_return']))
col3.metric("最大回撤", f"{metrics['max_drawdown']}%", 
            delta=f"{metrics['max_drawdown']}%", delta_color="inverse")
col4.metric("夏普比率", f"{metrics['sharpe_ratio']}", 
            delta=f"{metrics['sharpe_ratio']}", delta_color=format_color(metrics['sharpe_ratio']))

df = pd.DataFrame({
    'Date': pd.to_datetime(chart_data['dates']),
    'Strategy': chart_data['strategy'],
    'Benchmark': chart_data['benchmark'],
    'Drawdown': chart_data['drawdown']
})
df.set_index('Date', inplace=True)

# ==========================================
# 4. 图表 1：累计净值走势图
# ==========================================
st.markdown("### 📈 累计净值走势 (Strategy vs Benchmark)")

fig_nv = go.Figure()
# ========== 核心修改 3：更新图例名称 ==========
fig_nv.add_trace(go.Scatter(
    x=df.index, y=df['Strategy'], 
    name='策略净值 (动量+低PRP)', 
    line=dict(color='#ff4b4b', width=2)
))
fig_nv.add_trace(go.Scatter(
    x=df.index, y=df['Benchmark'], 
    name='深证成指基准', 
    line=dict(color='#3b82f6', width=2, dash='dot')
))

fig_nv.update_layout(
    hovermode="x unified",
    height=450,
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_nv, use_container_width=True)

# ==========================================
# 5. 图表 2：动态最大回撤图 (水下曲线)
# ==========================================
st.markdown("### 📉 动态最大回撤 (Drawdown)")

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=df.index, y=df['Drawdown'],
    name='回撤幅度 (%)',
    fill='tozeroy',
    line=dict(color='#f59e0b', width=1),
    fillcolor='rgba(245, 158, 11, 0.3)'
))

fig_dd.update_layout(
    hovermode="x unified",
    height=300,
    margin=dict(l=0, r=0, t=10, b=0),
    yaxis=dict(title='回撤百分比 (%)')
)
st.plotly_chart(fig_dd, use_container_width=True)
# ==========================================
# 6. 最新调仓股票池展示
# ==========================================
st.markdown("### 🛒 当前季度实盘持仓清单")

latest_holdings = data.get('latest_holdings', [])
if latest_holdings:
    # 将 Tushare 代码格式化，并附带雪球/东方财富的快捷搜索链接
    cols = st.columns(5)
    for idx, ts_code in enumerate(latest_holdings):
        symbol = ts_code.split('.')[0]
        # 根据后缀判断是 SH 还是 SZ
        market = "SH" if "SH" in ts_code else "SZ"
        with cols[idx % 5]:
            st.info(f"**{ts_code}**\n\n[查看行情](https://xueqiu.com/S/{market}{symbol})")
else:
    st.warning("暂无最新持仓数据，请等待回测脚本运行完毕。")
