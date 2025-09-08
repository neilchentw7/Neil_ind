import io
import math
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Neil 指標｜Excel 上傳分析", layout="wide")

st.markdown(
    """
# 📈 Neil 指標｜Excel 上傳分析

上傳你的回測/績效 Excel（包含逐時或逐筆的 **PROFIT / LOSS** 或 **NetPnL / Equity** 欄位），
本工具將自動計算：**Sharpe、Profit Factor、Calmar、Max/Avg Drawdown、回撤恢復時間、最大連續虧損次數**，
並輸出綜合單一分數 **Neil 指標**。

> 註：若沒有現成的報酬率欄位，請提供 **初始資金** 以便換算報酬率。
"""
)

# -----------------------------
# Sidebar: Parameters
# -----------------------------
st.sidebar.header("參數設定")
trading_days_per_year = st.sidebar.number_input(
    "一年交易天數 (Annualization)", min_value=1, max_value=366, value=252
)
periods_per_day = st.sidebar.number_input(
    "每日期間數 (例：逐時=24 或實際交易小時數)", min_value=1, max_value=240, value=24
)
risk_free_rate = st.sidebar.number_input(
    "無風險利率 (年化, 小數)", min_value=0.0, max_value=0.2, value=0.0, step=0.001
)

# -----------------------------
# File Uploader
# -----------------------------
uploaded = st.file_uploader("上傳 Excel 檔 (.xlsx)", type=["xlsx"])  

# -----------------------------
# Helper Functions
# -----------------------------

def annualization_factor(trading_days: int, periods_day: int) -> float:
    return math.sqrt(max(trading_days * periods_day, 1))


def compute_drawdowns(equity: pd.Series) -> dict:
    """計算回撤相關統計：序列、最大回撤、平均回撤、回撤區間、恢復時間等。
    回撤以『相對前高百分比』表示 (0~1)。
    """
    equity = pd.to_numeric(equity, errors="coerce").fillna(method="ffill").fillna(method="bfill").astype(float)
    peaks = equity.cummax()
    dd = (equity - peaks) / peaks  # 負值

    # 找出各回撤事件（由前高跌落直到回復新高）
    in_drawdown = False
    events = []  # list of dicts: {start, trough, end, depth}
    start_idx = None
    trough_idx = None
    trough_val = None

    for i, (e, p) in enumerate(zip(equity.values, peaks.values)):
        if e < p:  # 在回撤中
            if not in_drawdown:
                in_drawdown = True
                start_idx = i
                trough_idx = i
                trough_val = e
            else:
                if e < trough_val:
                    trough_val = e
                    trough_idx = i
        else:  # e == p: 非回撤狀態（等於新高，回撤結束）
            if in_drawdown:
                # 結束事件
                prev_peak = peaks[start_idx]
                depth = (trough_val - prev_peak) / prev_peak  # 負值
                events.append({
                    "start": start_idx,
                    "trough": trough_idx,
                    "end": i,
                    "depth": depth
                })
                in_drawdown = False
                start_idx = trough_idx = None
                trough_val = None

    # 若最後仍在回撤中，記錄未恢復事件（以當前最後值視為暫時結束點）
    if in_drawdown and start_idx is not None and trough_idx is not None:
        prev_peak = peaks[start_idx]
        depth = (trough_val - prev_peak) / prev_peak
        events.append({
            "start": start_idx,
            "trough": trough_idx,
            "end": len(equity) - 1,
            "depth": depth
        })

    depths = [e["depth"] for e in events] if events else []
    max_dd = abs(min(depths)) if depths else 0.0
    avg_dd = abs(np.mean(depths)) if depths else 0.0

    # 恢復時間：以最大回撤事件為主，計算 trough -> end 的期間長度
    max_event = None
    if events:
        max_event = min(events, key=lambda x: x["depth"])  # 最深(最負)者
        recovery_len = max(0, max_event["end"] - max_event["trough"])  # 期間長度
    else:
        recovery_len = 0

    return {
        "dd_series": dd,  # 負值
        "max_dd": max_dd,  # 正值 (比例)
        "avg_dd": avg_dd,  # 正值 (比例)
        "events": events,
        "max_event": max_event,
        "max_recovery_len": recovery_len
    }


def longest_losing_streak(pnl: pd.Series) -> int:
    streak = 0
    max_streak = 0
    for v in pd.to_numeric(pnl, errors="coerce").fillna(0).values:
        if v < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def safe_profit_factor(pnl: pd.Series) -> float:
    pnl = pd.to_numeric(pnl, errors="coerce").fillna(0.0)
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()  # 轉為正值
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains) / float(losses)


def compute_sharpe(returns: pd.Series, ann_factor: float, rf_annual: float = 0.0) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if len(r) < 2:
        return 0.0
    # 將年化無風險利率換算到單期（近似：periods_per_year = ann_factor^2）
    periods_per_year = max(int(round(ann_factor ** 2)), 1)
    rf_per_period = rf_annual / periods_per_year
    excess = r - rf_per_period
    mean = excess.mean()
    std = excess.std(ddof=1)
    if std == 0:
        return 0.0
    return (mean / std) * ann_factor


def compute_calmar(equity: pd.Series, trading_days: int, periods_day: int, max_dd_ratio: float) -> float:
    # 年化報酬率：使用首末淨值換算 CAGR
    equity = pd.to_numeric(equity, errors="coerce").dropna()
    if equity.empty:
        return 0.0
    start_val = float(equity.iloc[0])
    end_val = float(equity.iloc[-1])
    if start_val <= 0 or end_val <= 0:
        return 0.0
    n_years = max((len(equity) / (trading_days * periods_day)), 1e-9)
    cagr = (end_val / start_val) ** (1.0 / n_years) - 1.0
    if max_dd_ratio <= 0:
        return float("inf") if cagr > 0 else 0.0
    return cagr / max_dd_ratio


def compute_neil(
    sharpe: float,
    pf: float,
    max_dd: float,
    avg_dd: float,
    max_losing_streak: int,
    max_recov_len: int,
    total_len: int,
) -> float:
    # 恢復時間比例（0~1）
    recov_ratio = (max_recov_len / total_len) if total_len > 0 else 0.0
    denom = (1.0 + math.sqrt(max(max_dd * avg_dd, 0.0))) * (1.0 + max_losing_streak) * (1.0 + recov_ratio)
    if denom == 0:
        return 0.0
    # Profit Factor 可能為 inf，做個上限裁切避免溢位
    pf_capped = min(pf, 100.0)
    return (sharpe * pf_capped) / denom

# ====== 自動欄位偵測 ======
COMMON_ALIASES = {
    "timestamp": ["timestamp", "time", "datetime", "date", "日期", "時間"],
    "profit": ["profit", "gross profit", "profit_gross", "win", "獲利", "盈利"],
    "loss": ["loss", "gross loss", "loss_gross", "lose", "虧損"],
    "netpnl": ["netpnl", "pnl", "net pnl", "net_profit", "net", "盈虧", "淨損益"],
    "equity": ["equity", "balance", "networth", "nav", "淨值", "資金曲線"],
}


def guess_column(df: pd.DataFrame, key: str) -> str:
    """根據常見別名自動偵測欄位，回傳實際欄位名稱或 "<無>"。大小寫不敏感。"""
    if df is None or df.empty:
        return "<無>"
    aliases = COMMON_ALIASES.get(key, [])
    cols = list(df.columns.astype(str))
    lowmap = {c.lower(): c for c in cols}

    # 完整比對
    for a in aliases:
        if a.lower() in lowmap:
            return lowmap[a.lower()]

    # 模糊比對（包含關鍵字）
    for c in cols:
        lc = c.lower()
        for a in aliases:
            a = a.lower()
            if a in lc:
                return c
    return "<無>"

# -----------------------------
# Main
# -----------------------------
if uploaded is not None:
    try:
        # 讀取所有工作表
        xls = pd.ExcelFile(uploaded)
        sheets = xls.sheet_names
        st.success(f"已載入工作表：{sheets}")
        default_idx = sheets.index("Hourly Period Analysis") if "Hourly Period Analysis" in sheets else 0
        sheet_name = st.selectbox(
            "選擇要分析的工作表 (預設嘗試 'Hourly Period Analysis')",
            options=sheets,
            index=min(default_idx, len(sheets) - 1),
        )
        df = pd.read_excel(uploaded, sheet_name=sheet_name)

        st.subheader("原始資料預覽")
        st.dataframe(df.head(20), use_container_width=True)

        # 欄位對應設定（自動偵測 + 可覆寫）
        st.subheader("欄位對應設定")
        col_options = ["<無>"] + list(df.columns.astype(str))

        autod_ts = guess_column(df, "timestamp")
        autod_profit = guess_column(df, "profit")
        autod_loss = guess_column(df, "loss")
        autod_net = guess_column(df, "netpnl")
        autod_equity = guess_column(df, "equity")

        def opt_index(name: str) -> int:
            return col_options.index(name) if name in col_options else 0

        col_timestamp = st.selectbox(
            "時間欄位（可選，用於圖表 x 軸）",
            options=col_options,
            index=opt_index(autod_ts),
        )
        col_profit = st.selectbox(
            "PROFIT 欄位（可選）",
            options=col_options,
            index=opt_index(autod_profit if autod_profit != "<無>" else ("PROFIT" if "PROFIT" in df.columns else "<無>")),
        )
        col_loss = st.selectbox(
            "LOSS 欄位（可選）",
            options=col_options,
            index=opt_index(autod_loss if autod_loss != "<無>" else ("LOSS" if "LOSS" in df.columns else "<無>")),
        )
        col_net = st.selectbox(
            "NetPnL/盈虧 欄位（可選，若提供則優先使用）",
            options=col_options,
            index=opt_index(autod_net if autod_net != "<無>" else ("NetPnL" if "NetPnL" in df.columns else "<無>")),
        )
        col_equity = st.selectbox(
            "Equity/淨值 欄位（可選，若未提供則用初始資金累加 NetPnL 建立）",
            options=col_options,
            index=opt_index(autod_equity if autod_equity != "<無>" else ("Equity" if "Equity" in df.columns else "<無>")),
        )

        init_capital = st.number_input(
            "初始資金（若未提供 Equity，將以此 + 累計盈虧建立）",
            min_value=0.0,
            value=1_000_000.0,
            step=10_000.0,
            format="%.2f",
        )

        # 準備時間索引
        if col_timestamp != "<無>" and col_timestamp in df.columns:
            idx = pd.to_datetime(df[col_timestamp], errors="coerce")
        else:
            idx = pd.RangeIndex(start=0, stop=len(df), step=1)

        # 計算 NetPnL
        if col_net != "<無>" and col_net in df.columns:
            pnl = pd.to_numeric(df[col_net], errors="coerce").fillna(0.0)
        else:
            prof = (
                pd.to_numeric(df[col_profit], errors="coerce").fillna(0.0)
                if (col_profit != "<無>" and col_profit in df.columns)
                else pd.Series(np.zeros(len(df)))
            )
            loss = (
                pd.to_numeric(df[col_loss], errors="coerce").fillna(0.0)
                if (col_loss != "<無>" and col_loss in df.columns)
                else pd.Series(np.zeros(len(df)))
            )
            pnl = prof - loss

        pnl.index = idx

        # 建立 Equity
        if col_equity != "<無>" and col_equity in df.columns:
            equity = pd.to_numeric(df[col_equity], errors="coerce").fillna(method="ffill").fillna(method="bfill")
        else:
            equity = pd.Series(init_capital + pnl.cumsum(), index=idx)

        # 報酬率序列（以 Equity 推估）：使用 pct_change，若不可得則以 pnl/初始資金 近似
        if equity.isna().all():
            returns = pnl.astype(float) / max(init_capital, 1.0)
        else:
            returns = pd.Series(equity).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 基本統計
        ann_fac = annualization_factor(trading_days_per_year, periods_per_day)
        sharpe = compute_sharpe(returns, ann_fac, risk_free_rate)
        pf = safe_profit_factor(pnl)

        dd_stats = compute_drawdowns(equity)
        max_dd = dd_stats["max_dd"]  # 比例
        avg_dd = dd_stats["avg_dd"]  # 比例
        max_losing = longest_losing_streak(pnl)
        calmar = compute_calmar(equity, trading_days_per_year, periods_per_day, max_dd)

        max_recov_len = dd_stats["max_recovery_len"]
        total_len = len(equity)
        neil = compute_neil(
            sharpe, pf, max_dd, avg_dd, max_losing, max_recov_len, total_len
        )

        # -----------------------------
        # KPI Cards
        # -----------------------------
        kpi_cols = st.columns(6)
        kpi_cols[0].metric("Sharpe", f"{sharpe:.3f}")
        kpi_cols[1].metric("Profit Factor", ("∞" if math.isinf(pf) else f"{pf:.3f}"))
        kpi_cols[2].metric("Calmar", ("∞" if math.isinf(calmar) else f"{calmar:.3f}"))
        kpi_cols[3].metric("Max DD", f"{max_dd*100:.2f}%")
        kpi_cols[4].metric("Avg DD", f"{avg_dd*100:.2f}%")
        kpi_cols[5].metric("Neil 指標", f"{neil:.3f}")

        st.caption(
            f"最大連續虧損次數：{max_losing}｜最大回撤恢復時間長度：{max_recov_len}（佔比 {(max_recov_len/total_len*100 if total_len>0 else 0):.2f}% ）"
        )

        # -----------------------------
        # Charts (matplotlib，單圖單色，遵循平台限制)
        # -----------------------------
        st.subheader("資金淨值曲線 (Equity Curve)")
        fig1, ax1 = plt.subplots()
        ax1.plot(equity.index, equity.values)
        ax1.set_xlabel("Index/Time")
        ax1.set_ylabel("Equity")
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

        st.subheader("回撤曲線 (Drawdown)")
        fig2, ax2 = plt.subplots()
        ax2.plot(dd_stats["dd_series"].index, dd_stats["dd_series"].values)
        ax2.set_xlabel("Index/Time")
        ax2.set_ylabel("Drawdown (ratio)")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

        st.subheader("盈虧分佈 (PnL Histogram)")
        fig3, ax3 = plt.subplots()
        ax3.hist(pnl.values, bins=50)
        ax3.set_xlabel("PnL per period")
        ax3.set_ylabel("Frequency")
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)

        # -----------------------------
        # Summary Table
        # -----------------------------
        st.subheader("指標總結表")
        summary = pd.DataFrame(
            {
                "Sharpe": [sharpe],
                "ProfitFactor": [pf],
                "Calmar": [calmar],
                "MaxDrawdown(%)": [max_dd * 100],
                "AvgDrawdown(%)": [avg_dd * 100],
                "MaxLosingStreak": [max_losing],
                "MaxDD_RecoveryLen": [max_recov_len],
                "RecoveryLen_Ratio(%)": [
                    (max_recov_len / total_len * 100 if total_len > 0 else 0.0)
                ],
                "Neil": [neil],
            }
        )
        st.dataframe(summary, use_container_width=True)

        # 下載結果
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Raw", index=False)
            pd.DataFrame(pnl, columns=["NetPnL"]).to_excel(writer, sheet_name="NetPnL")
            pd.DataFrame(equity, columns=["Equity"]).to_excel(writer, sheet_name="Equity")
            pd.DataFrame(dd_stats["dd_series"], columns=["Drawdown"]).to_excel(
                writer, sheet_name="Drawdown"
            )
            summary.to_excel(writer, sheet_name="Summary", index=False)
        st.download_button(
            "下載分析結果 Excel",
            data=out.getvalue(),
            file_name="neil_indicator_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        with st.expander("Neil 指標公式與說明"):
            st.markdown(
                r"""
**Neil 指標**：

\[
\text{Neil} = \frac{\text{Sharpe} \times \text{ProfitFactor}}{\bigl(1 + \sqrt{\text{MaxDD} \times \text{AvgDD}}\bigr) \times (1 + \text{MaxLosingStreak}) \times (1 + \text{RecoveryRatio})}
\]

- **MaxDD/AvgDD**：以相對前高的比例（0~1）度量。
- **RecoveryRatio**：最大回撤事件的恢復期長度 / 全部期間長度。
- **Sharpe**：以 period 報酬率年化（年化係數 = \(\sqrt{\text{trading\_days} \times \text{periods\_per\_day}}\)）。
- **ProfitFactor**：\(\sum Gains / \sum |Losses|\)。

> ProfitFactor 為無窮大（無虧損）時，為避免數值失控，此實作上限裁切為 100。
                """
            )

    except Exception as e:
        st.error(f"讀取或分析時發生錯誤：{e}")
else:
    st.info("請先上傳 Excel 檔再開始分析。支援 .xlsx，若遇欄位命名不同，請在『欄位對應設定』選擇對應欄位。")
