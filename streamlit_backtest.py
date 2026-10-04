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

st.title("📈 基本面量化策略监控塔")
st.markdown("基于 `Tushare` 数据源的核心财务因子轮动策略 (高 ROE + 高股息)")

# ==========================================
# 2. 数据读取与解析
# ==========================================
@st.cache_data
def load_data(file_path="backtest_result.json"):
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

# 使用带颜色的 delta 标识正负收益
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

# 将 JSON 图表数据转为 Pandas DataFrame 以便 Plotly 渲染
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
# 策略红线
fig_nv.add_trace(go.Scatter(
    x=df.index, y=df['Strategy'], 
    name='策略净值', 
    line=dict(color='#ff4b4b', width=2)
))
# 基准蓝线 (虚线)
fig_nv.add_trace(go.Scatter(
    x=df.index, y=df['Benchmark'], 
    name='沪深300基准', 
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
# 回撤面积图
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
