
import io
from datetime import datetime
import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="T212 Dashboard LT", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
:root {
    --bg: #0f1115;
    --panel: #161a22;
    --panel2: #1c2230;
    --line: #2d3648;
    --text: #ecf2ff;
    --muted: #9fb0c9;
    --green: #49d17d;
    --red: #ff6b6b;
    --gold: #f4c15d;
    --blue: #67b7ff;
}
.stApp {
    background: linear-gradient(180deg, #0d0f14 0%, #10151d 100%);
    color: var(--text);
}
section[data-testid="stSidebar"] {
    background: #11161e;
    border-right: 1px solid var(--line);
}
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, #141922 0%, #171e29 100%);
    border: 1px solid #334158;
    padding: 10px 12px;
    border-radius: 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.18);
}
.block-card {
    background: linear-gradient(180deg, #151b24 0%, #121820 100%);
    border: 1px solid #304058;
    border-radius: 20px;
    padding: 16px 18px;
    margin-bottom: 14px;
}
.small-note {
    color: #9fb0c9;
    font-size: 0.88rem;
}
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    margin: 0 0 10px 0;
}
hr { border-color: #2d3648; }
</style>
""", unsafe_allow_html=True)

# ---------- CONSTANTS ----------
BUY_ACTIONS = {"Market buy", "Limit buy"}
SELL_ACTIONS = {"Market sell", "Limit sell"}
DIVIDEND_ACTIONS = {"Dividend (Dividend)"}
CASHBACK_ACTIONS = {"Spending cashback"}
INTEREST_CASH_ACTIONS = {"Interest on cash"}
LENDING_ACTIONS = {"Lending interest"}
DEPOSIT_ACTIONS = {"Deposit"}
CARD_ACTIONS = {"Card debit"}

# ---------- HELPERS ----------
@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)

def fmt_money(v: float, currency: str = "EUR") -> str:
    try:
        return f"{currency_symbol(currency)}{v:,.2f}".replace(",", " ")
    except Exception:
        return f"{v:,.2f}"

def fmt_pct(v: float) -> str:
    return f"{v:.2f}%"

def currency_symbol(code: str) -> str:
    mapping = {"EUR": "€", "USD": "$", "GBP": "£"}
    return mapping.get(code, f"{code} ")

def pick_main_currency(df: pd.DataFrame) -> str:
    if "Currency (Total)" in df.columns:
        vals = df["Currency (Total)"].dropna().astype(str)
        if not vals.empty:
            return vals.mode().iloc[0]
    return "EUR"

def prepare_transactions(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "Time" in data.columns:
        data["Time"] = pd.to_datetime(data["Time"], errors="coerce")
    for col in [
        "No. of shares", "Price / share", "Exchange rate", "Result", "Total",
        "Withholding tax", "Stamp duty reserve tax", "Currency conversion fee",
        "French transaction tax"
    ]:
        if col in data.columns:
            data[col] = to_num(data[col])

    for col in ["Ticker", "Name", "Action", "Merchant name", "Merchant category"]:
        if col in data.columns:
            data[col] = data[col].fillna("")

    data["signed_shares"] = 0.0
    data.loc[data["Action"].isin(BUY_ACTIONS), "signed_shares"] = data["No. of shares"]
    data.loc[data["Action"].isin(SELL_ACTIONS), "signed_shares"] = -data["No. of shares"]

    data["fees_total"] = (
        data.get("Withholding tax", 0)
        + data.get("Stamp duty reserve tax", 0)
        + data.get("Currency conversion fee", 0)
        + data.get("French transaction tax", 0)
    )

    return data.sort_values("Time").reset_index(drop=True)

def build_positions(transactions: pd.DataFrame) -> pd.DataFrame:
    trades = transactions[transactions["Action"].isin(BUY_ACTIONS | SELL_ACTIONS)].copy()
    trades = trades[trades["Ticker"] != ""].copy()
    if trades.empty:
        return pd.DataFrame(columns=[
            "Ticker", "Name", "shares", "avg_cost", "cost_basis_open",
            "realized_pnl", "first_buy", "last_buy", "buy_trades", "sell_trades"
        ])

    rows = []
    for ticker, grp in trades.groupby("Ticker", dropna=True):
        grp = grp.sort_values("Time")
        shares = 0.0
        cost_basis_open = 0.0
        realized_pnl = 0.0
        buy_trades = 0
        sell_trades = 0
        buy_dates = []
        sell_dates = []
        last_name = grp["Name"].replace("", pd.NA).dropna().iloc[-1] if grp["Name"].replace("", pd.NA).dropna().any() else ticker

        for _, row in grp.iterrows():
            qty = float(row.get("No. of shares", 0) or 0)
            total = float(row.get("Total", 0) or 0)
            action = row["Action"]
            dt = row.get("Time")

            if qty <= 0:
                continue

            if action in BUY_ACTIONS:
                shares += qty
                cost_basis_open += total
                buy_trades += 1
                if pd.notna(dt):
                    buy_dates.append(dt)
            elif action in SELL_ACTIONS:
                if shares <= 0:
                    continue
                avg_cost_before = cost_basis_open / shares if shares else 0.0
                proceeds = total
                realized_pnl += proceeds - qty * avg_cost_before
                shares -= qty
                cost_basis_open -= qty * avg_cost_before
                sell_trades += 1
                if pd.notna(dt):
                    sell_dates.append(dt)
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
            "first_buy": min(buy_dates) if buy_dates else pd.NaT,
            "last_buy": max(buy_dates) if buy_dates else pd.NaT,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
        })

    positions = pd.DataFrame(rows)
    positions = positions[positions["shares"] > 0].copy()
    return positions.sort_values("cost_basis_open", ascending=False).reset_index(drop=True)

@st.cache_data(show_spinner=False, ttl=900)
def fetch_prices(tickers: tuple[str, ...]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["Ticker", "current_price", "previous_close", "currency"])
    out = []
    for ticker in tickers:
        current_price = 0.0
        previous_close = 0.0
        currency = ""
        try:
            info = yf.Ticker(ticker).fast_info
            current_price = (
                info.get("lastPrice")
                or info.get("last_price")
                or info.get("regularMarketPrice")
                or info.get("regular_market_price")
                or 0.0
            )
            previous_close = (
                info.get("previousClose")
                or info.get("previous_close")
                or current_price
                or 0.0
            )
            currency = info.get("currency") or ""
        except Exception:
            pass
        out.append({
            "Ticker": ticker,
            "current_price": float(current_price or 0.0),
            "previous_close": float(previous_close or 0.0),
            "currency": currency,
        })
    return pd.DataFrame(out)

def enrich_positions(positions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return positions.copy()
    merged = positions.merge(prices, on="Ticker", how="left")
    merged["current_price"] = merged["current_price"].fillna(0.0)
    merged["previous_close"] = merged["previous_close"].fillna(0.0)
    merged["market_value"] = merged["shares"] * merged["current_price"]
    merged["unrealized_pnl"] = merged["market_value"] - merged["cost_basis_open"]
    merged["unrealized_pnl_pct"] = merged.apply(
        lambda r: ((r["unrealized_pnl"] / r["cost_basis_open"]) * 100) if r["cost_basis_open"] else 0.0,
        axis=1,
    )
    merged["day_change_value"] = (merged["current_price"] - merged["previous_close"]) * merged["shares"]
    total_mv = merged["market_value"].sum()
    merged["portfolio_weight_pct"] = merged["market_value"] / total_mv * 100 if total_mv else 0.0
    return merged.sort_values("market_value", ascending=False).reset_index(drop=True)

def summarize(transactions: pd.DataFrame, positions: pd.DataFrame, currency: str) -> dict:
    deposits = transactions.loc[transactions["Action"].isin(DEPOSIT_ACTIONS), "Total"].sum()
    buys = transactions.loc[transactions["Action"].isin(BUY_ACTIONS), "Total"].sum()
    sells = transactions.loc[transactions["Action"].isin(SELL_ACTIONS), "Total"].sum()
    dividends = transactions.loc[transactions["Action"].isin(DIVIDEND_ACTIONS), "Total"].sum()
    cashback = transactions.loc[transactions["Action"].isin(CASHBACK_ACTIONS), "Total"].sum()
    interest_cash = transactions.loc[transactions["Action"].isin(INTEREST_CASH_ACTIONS), "Total"].sum()
    lending_interest = transactions.loc[transactions["Action"].isin(LENDING_ACTIONS), "Total"].sum()
    card_spend = transactions.loc[transactions["Action"].isin(CARD_ACTIONS), "Total"].sum()
    fees = transactions["fees_total"].sum()

    market_value = positions["market_value"].sum() if "market_value" in positions else 0.0
    open_cost = positions["cost_basis_open"].sum() if "cost_basis_open" in positions else 0.0
    realized_pnl = positions["realized_pnl"].sum() if "realized_pnl" in positions else 0.0
    unrealized_pnl = positions["unrealized_pnl"].sum() if "unrealized_pnl" in positions else 0.0
    day_change = positions["day_change_value"].sum() if "day_change_value" in positions else 0.0

    # CSV net cash remaining after all ledger activity
    end_cash_balance_est = (
        deposits + sells + dividends + cashback + interest_cash + lending_interest + card_spend - buys
    )

    total_return = realized_pnl + unrealized_pnl + dividends + cashback + interest_cash + lending_interest - fees
    total_return_pct_on_deposits = (total_return / deposits * 100) if deposits else 0.0

    # Pot labels are estimates because CSV has no direct pot balance columns
    main_pot_est = market_value + max(end_cash_balance_est, 0.0)
    spending_pot_est = max(end_cash_balance_est, 0.0)

    sold_positions_count = len(transactions[transactions["Action"].isin(SELL_ACTIONS)]["Ticker"].replace("", pd.NA).dropna().unique())
    active_positions_count = len(positions)

    return {
        "currency": currency,
        "deposits": deposits,
        "buys": buys,
        "sells": sells,
        "dividends": dividends,
        "cashback": cashback,
        "interest_cash": interest_cash,
        "lending_interest": lending_interest,
        "card_spend": abs(card_spend),
        "fees": fees,
        "market_value": market_value,
        "open_cost": open_cost,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "day_change": day_change,
        "end_cash_balance_est": end_cash_balance_est,
        "main_pot_est": main_pot_est,
        "spending_pot_est": spending_pot_est,
        "total_return": total_return,
        "total_return_pct_on_deposits": total_return_pct_on_deposits,
        "active_positions_count": active_positions_count,
        "sold_positions_count": sold_positions_count,
        "trades_count": len(transactions[transactions["Action"].isin(BUY_ACTIONS | SELL_ACTIONS)]),
        "date_min": transactions["Time"].min(),
        "date_max": transactions["Time"].max(),
    }

def monthly_breakdown(transactions: pd.DataFrame) -> pd.DataFrame:
    x = transactions.copy()
    x["month"] = x["Time"].dt.to_period("M").astype(str)
    def sum_by(actions):
        return lambda s: s[x.loc[s.index, "Action"].isin(actions)].sum()
    out = x.groupby("month", dropna=True).agg(
        deposits=("Total", sum_by(DEPOSIT_ACTIONS)),
        buys=("Total", sum_by(BUY_ACTIONS)),
        sells=("Total", sum_by(SELL_ACTIONS)),
        dividends=("Total", sum_by(DIVIDEND_ACTIONS)),
        cashback=("Total", sum_by(CASHBACK_ACTIONS)),
        interest_cash=("Total", sum_by(INTEREST_CASH_ACTIONS)),
        lending_interest=("Total", sum_by(LENDING_ACTIONS)),
        card_spend=("Total", sum_by(CARD_ACTIONS)),
    ).reset_index()
    out["net_cash_flow"] = (
        out["deposits"] + out["sells"] + out["dividends"] + out["cashback"] +
        out["interest_cash"] + out["lending_interest"] + out["card_spend"] - out["buys"]
    )
    return out

def sold_results(transactions: pd.DataFrame) -> pd.DataFrame:
    trades = transactions[transactions["Action"].isin(BUY_ACTIONS | SELL_ACTIONS)].copy()
    trades = trades[trades["Ticker"] != ""].copy()
    rows = []
    for ticker, grp in trades.groupby("Ticker", dropna=True):
        grp = grp.sort_values("Time")
        shares = 0.0
        cost_basis = 0.0
        realized = 0.0
        sold_qty = 0.0
        sell_value = 0.0
        name = grp["Name"].replace("", pd.NA).dropna().iloc[-1] if grp["Name"].replace("", pd.NA).dropna().any() else ticker
        for _, row in grp.iterrows():
            qty = float(row.get("No. of shares", 0) or 0)
            total = float(row.get("Total", 0) or 0)
            if row["Action"] in BUY_ACTIONS:
                shares += qty
                cost_basis += total
            else:
                if shares <= 0:
                    continue
                avg = cost_basis / shares if shares else 0
                realized += total - qty * avg
                sold_qty += qty
                sell_value += total
                shares -= qty
                cost_basis -= qty * avg
                if abs(shares) < 1e-9:
                    shares = 0.0
                    cost_basis = 0.0
        if sold_qty > 0:
            rows.append({
                "Ticker": ticker,
                "Name": name,
                "Parduota vnt.": sold_qty,
                "Pardavimo suma": sell_value,
                "Realizuotas P/L": realized,
                "Statusas": "Pelninga" if realized >= 0 else "Nuostolinga",
                "Dar portfelyje": "Taip" if shares > 0 else "Ne",
            })
    return pd.DataFrame(rows).sort_values("Realizuotas P/L", ascending=False) if rows else pd.DataFrame()

def merchant_expenses(transactions: pd.DataFrame) -> pd.DataFrame:
    x = transactions[transactions["Action"].isin(CARD_ACTIONS)].copy()
    if x.empty:
        return pd.DataFrame(columns=["Merchant name", "Merchant category", "Total"])
    x["Spend abs"] = x["Total"].abs()
    out = x.groupby(["Merchant name", "Merchant category"], dropna=False)["Spend abs"].sum().reset_index()
    out = out.sort_values("Spend abs", ascending=False).rename(columns={"Spend abs": "Išleista"})
    return out

def render_metric_row(items):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            st.metric(item["label"], item["value"], item.get("delta"))

# ---------- DATA SOURCE ----------
with st.sidebar:
    st.markdown("## 📊 T212 Dashboard")
    st.markdown('<div class="small-note">Finansų analitiko stiliaus suvestinė iš tavo CSV</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Įkelk CSV", type=["csv"])
    use_demo = st.toggle("Naudoti tavo įkeltą demo CSV", value=True)
    refresh = st.button("🔄 Atnaujinti kainas")
    page = st.radio(
        "Meniu",
        ["Apžvalga", "Turimos akcijos", "Prekyba", "Pajamos", "Išlaidos", "Potai ir pinigai", "CSV diagnostika"],
        index=0,
    )
    st.markdown("---")
    st.caption("Pastaba: `Main Pot` ir `Spending Pot` šiame faile nėra tiesiogiai pateikti, todėl rodau aiškiai pažymėtas estimacijas iš sandorių srautų.")

example_path = "from_2025-09-09_to_2025-12-31_MTc3Njc4ODMyNTI5OQ.csv"

file_bytes = None
if uploaded is not None:
    file_bytes = uploaded.getvalue()
elif use_demo:
    try:
        with open(example_path, "rb") as f:
            file_bytes = f.read()
    except FileNotFoundError:
        pass

if file_bytes is None:
    st.info("Įkelk CSV failą kairėje. Gali naudoti ir šitą patį T212 / Revolut tipo eksportą.")
    st.stop()

raw = load_csv(file_bytes)
transactions = prepare_transactions(raw)
currency = pick_main_currency(transactions)
positions_base = build_positions(transactions)

if refresh:
    fetch_prices.clear()

prices = fetch_prices(tuple(sorted(positions_base["Ticker"].dropna().unique().tolist())))
positions = enrich_positions(positions_base, prices)
summary = summarize(transactions, positions, currency)
monthly = monthly_breakdown(transactions)
sold_df = sold_results(transactions)
expenses_df = merchant_expenses(transactions)

date_min = summary["date_min"]
date_max = summary["date_max"]
date_text = f"{date_min.date()} → {date_max.date()}" if pd.notna(date_min) and pd.notna(date_max) else "N/A"

# ---------- HEADER ----------
st.markdown(f"""
<div class="block-card">
    <div style="display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap;">
        <div style="font-size:1.5rem; font-weight:800;">📚 {page}</div>
        <div class="small-note"><b>{date_text}</b> &nbsp; | &nbsp; Sandorių: {len(transactions)} &nbsp; | &nbsp; Akcijų: {summary["trades_count"]} &nbsp; | &nbsp; Portfelyje: {summary["active_positions_count"]}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- PAGES ----------
if page == "Apžvalga":
    render_metric_row([
        {"label": "MAIN POT (est.)", "value": fmt_money(summary["main_pot_est"], currency)},
        {"label": "SPENDING POT (est.)", "value": fmt_money(summary["spending_pot_est"], currency)},
        {"label": "Šiuo metu akcijose", "value": fmt_money(summary["market_value"], currency)},
        {"label": "Realizuotas P/L", "value": fmt_money(summary["realized_pnl"], currency)},
        {"label": "Nerealizuotas P/L", "value": fmt_money(summary["unrealized_pnl"], currency)},
        {"label": "Bendra grąža", "value": fmt_money(summary["total_return"], currency), "delta": fmt_pct(summary["total_return_pct_on_deposits"])},
    ])

    st.markdown("### Pagrindinė suvestinė")
    render_metric_row([
        {"label": "Įnešta iš viso", "value": fmt_money(summary["deposits"], currency)},
        {"label": "Atvira savikaina", "value": fmt_money(summary["open_cost"], currency)},
        {"label": "Pasyvios pajamos", "value": fmt_money(summary["dividends"] + summary["cashback"] + summary["interest_cash"] + summary["lending_interest"], currency)},
        {"label": "Kortele išleista", "value": fmt_money(summary["card_spend"], currency)},
    ])

    c1, c2 = st.columns([2, 1])
    with c1:
        fig = go.Figure()
        if not sold_df.empty and "Realizuotas P/L" in sold_df.columns:
            sold_time = transactions[transactions["Action"].isin(SELL_ACTIONS)].copy()
            if not sold_time.empty:
                sold_time = sold_time.sort_values("Time")
                # approximate cumulative realized pnl by sell event using average cost engine
                cum_rows = []
                shares_book = {}
                cost_book = {}
                cum = 0.0
                trade_rows = transactions[transactions["Action"].isin(BUY_ACTIONS | SELL_ACTIONS)].copy().sort_values("Time")
                for _, row in trade_rows.iterrows():
                    ticker = row["Ticker"]
                    if not ticker:
                        continue
                    shares_book.setdefault(ticker, 0.0)
                    cost_book.setdefault(ticker, 0.0)
                    qty = float(row.get("No. of shares", 0) or 0)
                    total = float(row.get("Total", 0) or 0)
                    if row["Action"] in BUY_ACTIONS:
                        shares_book[ticker] += qty
                        cost_book[ticker] += total
                    else:
                        if shares_book[ticker] <= 0:
                            continue
                        avg = cost_book[ticker] / shares_book[ticker] if shares_book[ticker] else 0.0
                        pnl = total - qty * avg
                        cum += pnl
                        shares_book[ticker] -= qty
                        cost_book[ticker] -= qty * avg
                        cum_rows.append({"Time": row["Time"], "Kumuliatyvus realizuotas P/L": cum})
                if cum_rows:
                    cdf = pd.DataFrame(cum_rows)
                    fig = px.area(cdf, x="Time", y="Kumuliatyvus realizuotas P/L", title="Kumuliatyvus realizuotas P/L laike")
                    fig.update_layout(template="plotly_dark", paper_bgcolor="#11161e", plot_bgcolor="#11161e")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Dar nėra pardavimo kreivės duomenų.")
    with c2:
        if not sold_df.empty:
            pos = (sold_df["Realizuotas P/L"] >= 0).sum()
            neg = (sold_df["Realizuotas P/L"] < 0).sum()
        else:
            pos = 0
            neg = 0
        st.markdown('<div class="block-card"><div class="section-title">📋 Greita analizė</div>', unsafe_allow_html=True)
        st.write(f"**Pelningos parduotos pozicijos:** {pos}")
        st.write(f"**Nuostolingos parduotos pozicijos:** {neg}")
        st.write(f"**Atviros pozicijos dabar:** {summary['active_positions_count']}")
        st.write(f"**Dienos pokytis pagal kainas:** {fmt_money(summary['day_change'], currency)}")
        st.markdown('</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        if not positions.empty:
            fig2 = px.pie(positions, names="Ticker", values="market_value", hole=0.48, title="Portfelio sudėtis pagal dabartinę vertę")
            fig2.update_layout(template="plotly_dark", paper_bgcolor="#11161e", plot_bgcolor="#11161e")
            st.plotly_chart(fig2, use_container_width=True)
    with c4:
        if not monthly.empty:
            fig3 = px.bar(
                monthly,
                x="month",
                y=["deposits", "buys", "sells", "card_spend"],
                barmode="group",
                title="Mėnesiniai srautai",
            )
            fig3.update_layout(template="plotly_dark", paper_bgcolor="#11161e", plot_bgcolor="#11161e")
            st.plotly_chart(fig3, use_container_width=True)

elif page == "Turimos akcijos":
    st.markdown("### Dabartinės turimos pozicijos")
    if positions.empty:
        st.warning("Atvirų pozicijų nerasta.")
    else:
        tickers = ["Visi"] + positions["Ticker"].tolist()
        chosen = st.selectbox("Filtruok pagal tickerį", tickers, index=0)
        show = positions.copy()
        if chosen != "Visi":
            show = show[show["Ticker"] == chosen].copy()

        table = show[[
            "Ticker", "Name", "shares", "first_buy", "last_buy", "buy_trades", "avg_cost",
            "current_price", "cost_basis_open", "market_value", "unrealized_pnl",
            "unrealized_pnl_pct", "realized_pnl", "portfolio_weight_pct", "currency"
        ]].copy()
        table = table.rename(columns={
            "shares": "Kiekis",
            "first_buy": "Pirmas pirkimas",
            "last_buy": "Paskutinis pirkimas",
            "buy_trades": "Pirkimų sk.",
            "avg_cost": "Vid. pirkimo kaina",
            "current_price": "Dabartinė kaina",
            "cost_basis_open": "Pirkimo vertė",
            "market_value": "Dabartinė vertė",
            "unrealized_pnl": "Pelnas/nuostolis dabar",
            "unrealized_pnl_pct": "P/L %",
            "realized_pnl": "Realizuotas P/L",
            "portfolio_weight_pct": "Svoris portfelyje %",
            "currency": "Valiuta",
        })
        st.dataframe(table, use_container_width=True, hide_index=True)

        fig = px.bar(
            show.sort_values("unrealized_pnl", ascending=False),
            x="Ticker",
            y="unrealized_pnl",
            color="unrealized_pnl",
            title="Atvirų pozicijų P/L",
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#11161e", plot_bgcolor="#11161e")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Prekyba":
    st.markdown("### Pirkimai, pardavimai ir istorija")
    c1, c2, c3 = st.columns(3)
    c1.metric("Pirkimų suma", fmt_money(summary["buys"], currency))
    c2.metric("Pardavimų suma", fmt_money(summary["sells"], currency))
    c3.metric("Realizuotas P/L", fmt_money(summary["realized_pnl"], currency))

    if not sold_df.empty:
        st.markdown("#### Parduotos pozicijos")
        st.dataframe(sold_df, use_container_width=True, hide_index=True)

    trade_view = transactions[transactions["Action"].isin(BUY_ACTIONS | SELL_ACTIONS)].copy()
    if not trade_view.empty:
        choose_action = st.multiselect(
            "Rodyti veiksmus",
            sorted(trade_view["Action"].unique().tolist()),
            default=sorted(trade_view["Action"].unique().tolist())
        )
        trade_view = trade_view[trade_view["Action"].isin(choose_action)].copy()
        st.markdown("#### Sandorių juosta")
        st.dataframe(
            trade_view[["Time", "Action", "Ticker", "Name", "No. of shares", "Price / share", "Total", "Currency (Total)"]],
            use_container_width=True,
            hide_index=True
        )

elif page == "Pajamos":
    st.markdown("### Dividendai, cashback, palūkanos")
    render_metric_row([
        {"label": "Dividendai", "value": fmt_money(summary["dividends"], currency)},
        {"label": "Cashback", "value": fmt_money(summary["cashback"], currency)},
        {"label": "Cash palūkanos", "value": fmt_money(summary["interest_cash"], currency)},
        {"label": "Lending palūkanos", "value": fmt_money(summary["lending_interest"], currency)},
    ])

    income_actions = DIVIDEND_ACTIONS | CASHBACK_ACTIONS | INTEREST_CASH_ACTIONS | LENDING_ACTIONS
    income_df = transactions[transactions["Action"].isin(income_actions)].copy()
    if not income_df.empty:
        income_month = income_df.copy()
        income_month["month"] = income_month["Time"].dt.to_period("M").astype(str)
        pivot = income_month.groupby(["month", "Action"])["Total"].sum().reset_index()
        fig = px.bar(pivot, x="month", y="Total", color="Action", barmode="group", title="Pajamos pagal mėnesį")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#11161e", plot_bgcolor="#11161e")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            income_df[["Time", "Action", "Ticker", "Name", "Total", "Currency (Total)", "Notes"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Pajamų įrašų nerasta.")

elif page == "Išlaidos":
    st.markdown("### Kortelės išlaidos")
    st.metric("Iš viso kortele išleista", fmt_money(summary["card_spend"], currency))
    if not expenses_df.empty:
        c1, c2 = st.columns([1.2, 1.8])
        with c1:
            top = expenses_df.head(10)
            fig = px.bar(top, x="Išleista", y="Merchant name", orientation="h", title="Top merchantai pagal išlaidas")
            fig.update_layout(template="plotly_dark", paper_bgcolor="#11161e", plot_bgcolor="#11161e")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.dataframe(expenses_df, use_container_width=True, hide_index=True)

        cat = transactions[transactions["Action"].isin(CARD_ACTIONS)].copy()
        if not cat.empty:
            cat["Abs total"] = cat["Total"].abs()
            cat2 = cat.groupby("Merchant category")["Abs total"].sum().reset_index().sort_values("Abs total", ascending=False)
            fig2 = px.pie(cat2, names="Merchant category", values="Abs total", hole=0.42, title="Išlaidos pagal kategoriją")
            fig2.update_layout(template="plotly_dark", paper_bgcolor="#11161e", plot_bgcolor="#11161e")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Kortelės išlaidų nerasta.")

elif page == "Potai ir pinigai":
    st.markdown("### Potų ir pinigų estimacijos")
    st.warning("Tavo CSV neturi tiesioginių `Main Pot` ar `Spending Pot` balansų stulpelių. Žemiau pateikti rodikliai yra skaičiuojami iš įplaukų, prekybos ir kortelės srautų, todėl jie yra **estimacijos**.")
    render_metric_row([
        {"label": "MAIN POT (est.)", "value": fmt_money(summary["main_pot_est"], currency)},
        {"label": "SPENDING POT (est.)", "value": fmt_money(summary["spending_pot_est"], currency)},
        {"label": "Galutinis cash balansas (est.)", "value": fmt_money(summary["end_cash_balance_est"], currency)},
        {"label": "Dabartinė akcijų vertė", "value": fmt_money(summary["market_value"], currency)},
    ])

    st.markdown("""
**Kaip skaičiuoju:**
- **Galutinis cash balansas (est.)** = įnešimai + pardavimai + dividendai + cashback + cash palūkanos + lending palūkanos + kortelės operacijos - pirkimai
- **SPENDING POT (est.)** = teigiama galutinio cash balanso dalis
- **MAIN POT (est.)** = dabartinė akcijų vertė + teigiama galutinio cash balanso dalis

Jei norėsi, galiu pridėti atskirą rankinį `Main Pot / Spending Pot` mapping'ą, jei turi kitą eksportą su tiesioginiais potų balansais.
""")

    if not monthly.empty:
        fig = px.line(monthly, x="month", y="net_cash_flow", markers=True, title="Mėnesinis netto cash flow")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#11161e", plot_bgcolor="#11161e")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(monthly, use_container_width=True, hide_index=True)

elif page == "CSV diagnostika":
    st.markdown("### CSV struktūra ir diagnostika")
    st.write("Šita skiltis naudinga, jei vėliau norėsi, kad dashboardas būtų dar tikslesnis.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Veiksmų tipai")
        st.dataframe(transactions["Action"].value_counts().rename_axis("Action").reset_index(name="Kiekis"), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("#### Stulpeliai")
        st.write(list(raw.columns))
    st.markdown("#### Žali duomenys")
    st.dataframe(raw.head(100), use_container_width=True, hide_index=True)
