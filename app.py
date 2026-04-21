
import io
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Akciju portfelio dashboard", layout="wide")

BUY_ACTIONS = {"Market buy", "Limit buy"}
SELL_ACTIONS = {"Market sell", "Limit sell"}
DIVIDEND_ACTIONS = {"Dividend (Dividend)"}
CASH_IN_ACTIONS = {"Deposit", "Spending cashback", "Interest on cash", "Lending interest"}
CASH_OUT_ACTIONS = {"Card debit"}

@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)

def prepare_transactions(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["Time"] = pd.to_datetime(data["Time"], errors="coerce")
    for col in ["No. of shares", "Price / share", "Total", "Exchange rate", "Result"]:
        if col in data.columns:
            data[col] = to_num(data[col])

    data["signed_shares"] = 0.0
    data.loc[data["Action"].isin(BUY_ACTIONS), "signed_shares"] = data["No. of shares"]
    data.loc[data["Action"].isin(SELL_ACTIONS), "signed_shares"] = -data["No. of shares"]

    data["cash_flow"] = 0.0
    data.loc[data["Action"].isin(CASH_IN_ACTIONS | DIVIDEND_ACTIONS), "cash_flow"] = data["Total"]
    data.loc[data["Action"].isin(CASH_OUT_ACTIONS), "cash_flow"] = data["Total"]

    data["trade_value"] = 0.0
    data.loc[data["Action"].isin(BUY_ACTIONS | SELL_ACTIONS), "trade_value"] = data["Total"]

    return data.sort_values("Time")

def build_positions(transactions: pd.DataFrame) -> pd.DataFrame:
    trades = transactions[transactions["Action"].isin(BUY_ACTIONS | SELL_ACTIONS)].copy()
    trades = trades[trades["Ticker"].notna()].copy()

    if trades.empty:
        return pd.DataFrame(columns=[
            "Ticker", "Name", "shares", "avg_cost", "cost_basis_open", "realized_pnl"
        ])

    rows = []
    for ticker, grp in trades.groupby("Ticker", dropna=True):
        grp = grp.sort_values("Time")
        shares = 0.0
        cost_basis_open = 0.0
        realized_pnl = 0.0
        last_name = grp["Name"].dropna().iloc[-1] if grp["Name"].notna().any() else ticker

        for _, row in grp.iterrows():
            qty = float(row["No. of shares"])
            total = float(row["Total"])
            action = row["Action"]

            if qty <= 0:
                continue

            if action in BUY_ACTIONS:
                shares += qty
                cost_basis_open += total
            elif action in SELL_ACTIONS:
                if shares <= 0:
                    continue
                avg_cost_before = cost_basis_open / shares if shares else 0.0
                proceeds = total
                realized_pnl += proceeds - qty * avg_cost_before
                shares -= qty
                cost_basis_open -= qty * avg_cost_before
                if abs(shares) < 1e-9:
                    shares = 0.0
                    cost_basis_open = 0.0

        avg_cost = cost_basis_open / shares if shares else 0.0

        rows.append({
            "Ticker": ticker,
            "Name": last_name,
            "shares": shares,
            "avg_cost": avg_cost,
            "cost_basis_open": cost_basis_open,
            "realized_pnl": realized_pnl,
        })

    positions = pd.DataFrame(rows)
    positions = positions[positions["shares"] > 0].copy()
    return positions.sort_values("cost_basis_open", ascending=False)

@st.cache_data(show_spinner=False, ttl=900)
def fetch_prices(tickers: tuple[str, ...]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["Ticker", "current_price", "currency", "previous_close"])

    out = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).fast_info
            current_price = (
                info.get("lastPrice")
                or info.get("last_price")
                or info.get("regularMarketPrice")
                or 0.0
            )
            previous_close = (
                info.get("previousClose")
                or info.get("previous_close")
                or current_price
                or 0.0
            )
            currency = info.get("currency") or ""
            out.append({
                "Ticker": ticker,
                "current_price": float(current_price or 0.0),
                "previous_close": float(previous_close or 0.0),
                "currency": currency,
            })
        except Exception:
            out.append({
                "Ticker": ticker,
                "current_price": 0.0,
                "previous_close": 0.0,
                "currency": "",
            })
    return pd.DataFrame(out)

def enrich_positions(positions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return positions

    merged = positions.merge(prices, on="Ticker", how="left")
    merged["current_price"] = merged["current_price"].fillna(0.0)
    merged["market_value"] = merged["shares"] * merged["current_price"]
    merged["unrealized_pnl"] = merged["market_value"] - merged["cost_basis_open"]
    merged["unrealized_pnl_pct"] = merged.apply(
        lambda r: (r["unrealized_pnl"] / r["cost_basis_open"] * 100) if r["cost_basis_open"] else 0.0,
        axis=1,
    )
    merged["day_change_value"] = (merged["current_price"] - merged["previous_close"].fillna(0.0)) * merged["shares"]
    return merged.sort_values("market_value", ascending=False)

def make_kpis(transactions: pd.DataFrame, positions: pd.DataFrame) -> dict:
    deposits = transactions.loc[transactions["Action"] == "Deposit", "Total"].sum()
    card_spend = transactions.loc[transactions["Action"] == "Card debit", "Total"].sum()
    dividends = transactions.loc[transactions["Action"].isin(DIVIDEND_ACTIONS), "Total"].sum()
    cashback = transactions.loc[transactions["Action"] == "Spending cashback", "Total"].sum()
    interest = transactions.loc[transactions["Action"].isin({"Interest on cash", "Lending interest"}), "Total"].sum()

    total_market_value = positions["market_value"].sum() if "market_value" in positions else 0.0
    total_cost_open = positions["cost_basis_open"].sum() if "cost_basis_open" in positions else 0.0
    total_unrealized = positions["unrealized_pnl"].sum() if "unrealized_pnl" in positions else 0.0
    total_realized = positions["realized_pnl"].sum() if "realized_pnl" in positions else 0.0
    total_day_change = positions["day_change_value"].sum() if "day_change_value" in positions else 0.0

    return {
        "deposits": deposits,
        "card_spend": card_spend,
        "dividends": dividends,
        "cashback": cashback,
        "interest": interest,
        "total_market_value": total_market_value,
        "total_cost_open": total_cost_open,
        "total_unrealized": total_unrealized,
        "total_realized": total_realized,
        "total_day_change": total_day_change,
        "open_positions": len(positions),
    }

def fmt_money(v: float) -> str:
    return f"{v:,.2f}".replace(",", " ")

st.title("Akciju portfelio dashboard")
st.caption("Ikelk Revolut tipo sandoriu CSV ir gauk atviras pozicijas, savikaina, realizuota bei nerealizuota P/L ir dabartines kainas per Yahoo Finance.")

with st.sidebar:
    st.header("Nustatymai")
    uploaded = st.file_uploader("Ikelk CSV faila", type=["csv"])
    use_example = st.toggle("Naudoti demo faila is projekto aplanko", value=False)
    hide_small = st.number_input("Nerodyti poziciju, kuriu verte mazesne nei", min_value=0.0, value=0.0, step=10.0)
    refresh = st.button("Atnaujinti kainas")

example_path = "from_2025-09-09_to_2025-12-31_MTc3Njc4ODMyNTI5OQ.csv"

file_bytes = None
if uploaded is not None:
    file_bytes = uploaded.getvalue()
elif use_example:
    try:
        with open(example_path, "rb") as f:
            file_bytes = f.read()
    except FileNotFoundError:
        st.info("Demo failas nerastas. Ikelk savo CSV.")
else:
    st.info("Pradek nuo CSV ikelimo. Tinka sandoriu eksportas su stulpeliais Action, Time, Ticker, No. of shares, Price / share, Total.")
    st.stop()

raw = load_csv(file_bytes)
transactions = prepare_transactions(raw)
positions_base = build_positions(transactions)

if refresh:
    fetch_prices.clear()

prices = fetch_prices(tuple(sorted(positions_base["Ticker"].dropna().unique().tolist())))
positions = enrich_positions(positions_base, prices)

if hide_small > 0:
    positions = positions[positions["market_value"] >= hide_small].copy()

kpis = make_kpis(transactions, positions)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Portfelio verte", fmt_money(kpis["total_market_value"]))
col2.metric("Atvira savikaina", fmt_money(kpis["total_cost_open"]))
col3.metric("Nerealizuotas P/L", fmt_money(kpis["total_unrealized"]))
col4.metric("Realizuotas P/L", fmt_money(kpis["total_realized"]))
col5.metric("Dienos pokytis", fmt_money(kpis["total_day_change"]))

col6, col7, col8, col9, col10 = st.columns(5)
col6.metric("Atviros pozicijos", str(kpis["open_positions"]))
col7.metric("Inesai", fmt_money(kpis["deposits"]))
col8.metric("Dividendai", fmt_money(kpis["dividends"]))
col9.metric("Cashback + palukanos", fmt_money(kpis["cashback"] + kpis["interest"]))
col10.metric("Korteles islaidos", fmt_money(kpis["card_spend"]))

tab1, tab2, tab3, tab4 = st.tabs(["Pozicijos", "Grafikai", "Cash flow", "Sandoriai"])

with tab1:
    if positions.empty:
        st.warning("Atviru poziciju nerasta.")
    else:
        table = positions.copy()
        table["weight_pct"] = table["market_value"] / table["market_value"].sum() * 100
        show_cols = [
            "Ticker", "Name", "shares", "avg_cost", "current_price", "cost_basis_open",
            "market_value", "unrealized_pnl", "unrealized_pnl_pct", "realized_pnl",
            "day_change_value", "weight_pct", "currency"
        ]
        st.dataframe(
            table[show_cols].rename(columns={
                "shares": "Kiekis",
                "avg_cost": "Vid. savikaina",
                "current_price": "Dab. kaina",
                "cost_basis_open": "Atvira savikaina",
                "market_value": "Rinkos verte",
                "unrealized_pnl": "Nerealizuotas P/L",
                "unrealized_pnl_pct": "Nerealizuotas P/L %",
                "realized_pnl": "Realizuotas P/L",
                "day_change_value": "Dienos pokytis",
                "weight_pct": "Svoris %",
                "currency": "Valiuta",
            }),
            use_container_width=True,
            hide_index=True,
        )

with tab2:
    if positions.empty:
        st.warning("Grafikams reikia bent vienos atviros pozicijos.")
    else:
        c1, c2 = st.columns(2)

        fig1 = px.pie(
            positions,
            names="Ticker",
            values="market_value",
            title="Portfelio sudetis pagal rinkos verte",
            hole=0.45,
        )
        c1.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(
            positions.sort_values("unrealized_pnl", ascending=False),
            x="Ticker",
            y="unrealized_pnl",
            title="Nerealizuotas P/L pagal pozicija",
            text_auto=".2s",
        )
        c2.plotly_chart(fig2, use_container_width=True)

        fig3 = px.bar(
            positions.sort_values("market_value", ascending=False),
            x="Ticker",
            y="market_value",
            title="Poziciju verte",
            text_auto=".2s",
        )
        st.plotly_chart(fig3, use_container_width=True)

with tab3:
    monthly = transactions.copy()
    monthly["month"] = monthly["Time"].dt.to_period("M").astype(str)
    monthly_summary = monthly.groupby("month", dropna=True).agg(
        deposit_sum=("Total", lambda s: s[monthly.loc[s.index, "Action"] == "Deposit"].sum()),
        dividends=("Total", lambda s: s[monthly.loc[s.index, "Action"].isin(DIVIDEND_ACTIONS)].sum()),
        interest=("Total", lambda s: s[monthly.loc[s.index, "Action"].isin({"Interest on cash", "Lending interest"})].sum()),
        cashback=("Total", lambda s: s[monthly.loc[s.index, "Action"] == "Spending cashback"].sum()),
        card_debit=("Total", lambda s: s[monthly.loc[s.index, "Action"] == "Card debit"].sum()),
        buy_total=("Total", lambda s: s[monthly.loc[s.index, "Action"].isin(BUY_ACTIONS)].sum()),
        sell_total=("Total", lambda s: s[monthly.loc[s.index, "Action"].isin(SELL_ACTIONS)].sum()),
    ).reset_index()

    st.dataframe(monthly_summary, use_container_width=True, hide_index=True)

    fig4 = px.bar(
        monthly_summary,
        x="month",
        y=["deposit_sum", "buy_total", "sell_total", "dividends", "cashback", "interest"],
        barmode="group",
        title="Menesine pinigu ir investiciju apyvarta",
    )
    st.plotly_chart(fig4, use_container_width=True)

with tab4:
    show_actions = st.multiselect(
        "Filtruoti veiksmus",
        options=sorted(transactions["Action"].dropna().unique().tolist()),
        default=sorted(transactions["Action"].dropna().unique().tolist()),
    )
    txn_view = transactions[transactions["Action"].isin(show_actions)].copy()
    txn_view = txn_view[[
        "Time", "Action", "Ticker", "Name", "No. of shares", "Price / share", "Total", "Currency (Total)"
    ]]
    st.dataframe(txn_view.sort_values("Time", ascending=False), use_container_width=True, hide_index=True)

st.divider()
st.markdown(
    """
**Pastabos**
- Dabartines kainos imamos per `yfinance`, todel kai kuriems ETF ar instrumentams gali reiketi birzos sufikso.
- Jei kuris nors tickeris neuzsikrauna, galima ideti rankine map lentele, pvz. `RR -> RR.L`.
- Jei noresi, galiu tau prideti sektorius, salis, dividend yield, benchmark palyginima ir istorini equity curve.
"""
)
