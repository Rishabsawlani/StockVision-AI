
import sys
from pathlib import Path

# Add project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import plotly.express as px
import pandas as pd

from src.ai.gemini_client import generate_stock_analysis

from src.analysis.analytics_engine import (
    build_stock_comparison,
    load_stock_prices
)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_stock_analysis(stock_metrics):
    return generate_stock_analysis(stock_metrics)


# Header
st.markdown(
    """
    <div>
        <h1 class="main-heading"> StockVision-AI</h1>
        <p class="main-subtitle">
            Historical Stock Analytics &amp; Sector Intelligence
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


st.markdown(
    """
    <style>

    /* Increase sidebar navigation font size */
    [data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 18px !important;
        font-weight: 600 !important;
    }

    /* Main content spacing */
    .block-container {
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1500px;
    }

    /* Add breathing room between charts */
    div[data-testid="stPlotlyChart"] {
        margin-top: 1rem;
        margin-bottom: 2rem;
    }

    /* Main dashboard heading */
    .main-heading {
        text-align: center;
        font-size: 3.8rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin-top: 10px;
        margin-bottom: 8px;
    }

    /* Dashboard subtitle */
    .main-subtitle {
        text-align: center;
        font-size: 1.25rem;
        color: #A9B8CE;
        margin-top: 0;
        margin-bottom: 20px;
    }

    /* Force all KPI cards to have equal height */
    [data-testid="stHorizontalBlock"] [data-testid="stMetric"] {
        height: 115px !important;
        min-height: 115px !important;
        box-sizing: border-box !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        padding: 16px !important;
    }

    /* Dashboard headings */
    h1 {
        letter-spacing: -1px;
        font-weight: 750;
    }

    h2, h3 {
        letter-spacing: -0.4px;
        font-weight: 650;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(
            135deg,
            #111827 0%,
            #172238 100%
        );
        border: 1px solid #263750;
        padding: 20px 22px;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
    }

    div[data-testid="stMetricLabel"] {
        color: #A9B8CE;
        font-size: 0.9rem;
    }

    div[data-testid="stMetricValue"] {
        color: #E8F0FF;
        font-weight: 700;
    }

    /* Subtle separators */
    hr {
        border-color: #263247;
    }

    /* Rounded chart containers */
    div[data-testid="stPlotlyChart"] {
        border: 1px solid #202D42;
        border-radius: 12px;
        padding: 8px;
        background: #0E1627;
    }
    </style>
    """,
    unsafe_allow_html=True
)





# Sidebar navigation
st.sidebar.title("Navigation")


page = st.sidebar.radio(
    "Navigation",
    [
        "Market Overview",
        "Stock Analysis",
        "Technical Analysis",
        "Sector Analysis",
        "Risk Analysis",
        "AI Stock Analyst"
    ]
)

st.sidebar.divider()

if st.sidebar.button("Refresh Latest Data 🔃", use_container_width=True):

    with st.sidebar.status("Updating stock market data...", expanded=True) as status:
        try:
            from src.data.update_market_data import update_stock_data

            update_stock_data()

            st.cache_data.clear()

            status.update(
                label="Stock data updated successfully!",
                state="complete",
                expanded=False
            )

            st.rerun()

        except Exception as error:
            status.update(
                label="Stock data update failed",
                state="error",
                expanded=True
            )

            st.sidebar.error(str(error))



# Load analytics and cache results for five minutes
@st.cache_data(ttl=300)
def load_stock_comparison():
    return build_stock_comparison()


@st.cache_data(ttl=300)
def load_all_stock_prices():
    prices = load_stock_prices()

    prices["date"] = pd.to_datetime(
        prices["date"]
    )

    return prices


def calculate_period_comparison(prices):
    """
    Recalculate stock performance and risk metrics
    using only the supplied historical price data.
    """

    results = []

    for ticker, stock_data in prices.groupby("ticker"):

        stock_data = stock_data.sort_values(
            "date"
        ).copy()

        stock_data = stock_data.dropna(
            subset=["close"]
        )

        if len(stock_data) < 2:
            continue

        first_close = stock_data["close"].iloc[0]
        latest_close = stock_data["close"].iloc[-1]

        cumulative_return = (
            (latest_close / first_close) - 1
        ) * 100

        daily_returns = (
            stock_data["close"].pct_change()
        )

        annualized_volatility = (
            daily_returns.std() * (252 ** 0.5) * 100
        )

        running_peak = (
            stock_data["close"].cummax()
        )

        drawdowns = (
            (stock_data["close"] / running_peak) - 1
        ) * 100

        maximum_drawdown = drawdowns.min()

        results.append(
            {
                "ticker": ticker,
                "sector": stock_data["sector"].iloc[0],
                "date": stock_data["date"].iloc[-1],
                "close": latest_close,
                "Cumulative_Return_Pct": cumulative_return,
                "Annualized_Volatility_Pct": annualized_volatility,
                "Maximum_Drawdown_Pct": maximum_drawdown,
            }
        )

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        "Cumulative_Return_Pct",
        ascending=False
    ).reset_index(drop=True)

    comparison["Return_Rank"] = (
        comparison["Cumulative_Return_Pct"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    return comparison



# Load historical prices for date filtering
prices = load_all_stock_prices()

# Find the available date range
latest_date = prices["date"].max()
earliest_date = prices["date"].min()

st.sidebar.divider()

st.sidebar.subheader("Analysis Period")

date_range = st.sidebar.selectbox(
    "Select historical period",
    [
        "1 Year",
        "3 Years",
        "All History",
        "Custom Range"
    ],
    index=2
)

if date_range == "1 Year":
    start_date = latest_date - pd.DateOffset(years=1)
    end_date = latest_date

elif date_range == "3 Years":
    start_date = latest_date - pd.DateOffset(years=3)
    end_date = latest_date

elif date_range == "Custom Range":
    selected_dates = st.sidebar.date_input(
        "Choose date range",
        value=(
            earliest_date.date(),
            latest_date.date()
        ),
        min_value=earliest_date.date(),
        max_value=latest_date.date()
    )

    if len(selected_dates) == 2:
        start_date = pd.Timestamp(selected_dates[0])
        end_date = pd.Timestamp(selected_dates[1])
    else:
        start_date = earliest_date
        end_date = latest_date

else:
    start_date = earliest_date
    end_date = latest_date

# Filter prices using the selected dates
filtered_prices = prices[
    (prices["date"] >= start_date)
    & (prices["date"] <= end_date)
].copy()


# Shared stock selection for Market Overview and Risk Analysis
available_tickers = sorted(prices["ticker"].unique())

if "shared_stock_selection" not in st.session_state:
    st.session_state["shared_stock_selection"] = available_tickers[:6]


# Shared single-stock selection for Stock Analysis and Technical Analysis
if "shared_single_stock" not in st.session_state:
    st.session_state["shared_single_stock"] = (
        available_tickers[0] if available_tickers else None
    )


# Keep only tickers that still exist in the database
st.session_state["shared_stock_selection"] = [
    ticker
    for ticker in st.session_state["shared_stock_selection"]
    if ticker in available_tickers
]

if len(st.session_state["shared_stock_selection"]) < 5:
    st.session_state["shared_stock_selection"] = available_tickers[:6]




# Render the selector only on the two comparison pages
if page in ["Market Overview", "Risk Analysis"]:


    st.markdown(
        "<div style='text-align:center;'>"
        "<h3>Stock Comparison</h3>"
        "<p>Select 5–10 stocks to compare across both pages</p>"
        "</div>",
        unsafe_allow_html=True
    )


    left_space, selector_col, right_space = st.columns([1, 3, 1])

    with selector_col:
        st.markdown("**Selected stocks**")

        selected_tickers = st.session_state["shared_stock_selection"]

        # Display selected stocks in a readable multi-row layout.
        chip_columns = st.columns(3)

        for index, ticker in enumerate(selected_tickers):
            with chip_columns[index % 3]:
                st.button(
                    f"✕  {ticker}",
                    key=f"remove_stock_{ticker}",
                    use_container_width=True,
                    on_click=lambda t=ticker: st.session_state.update({
                        "shared_stock_selection": [
                            stock
                            for stock in st.session_state["shared_stock_selection"]
                            if stock != t
                        ]
                    })
                )

        remaining_tickers = [
            ticker for ticker in available_tickers
            if ticker not in selected_tickers
        ]

        if len(selected_tickers) < 10:
            new_ticker = st.selectbox(
                "Add a stock",
                options=["Select a stock..."] + remaining_tickers,
                key="add_stock_selector"
            )

            if new_ticker != "Select a stock...":
                st.session_state["shared_stock_selection"].append(new_ticker)
                st.rerun()


    if len(selected_tickers) < 5:
        st.warning("Select at least 5 stocks to display the comparison.")



# Recalculate metrics for the selected period
comparison = calculate_period_comparison(
    filtered_prices
)

st.sidebar.caption(
    f"{start_date:%d %b %Y} to {end_date:%d %b %Y}"
)



if page == "Market Overview":

    st.header("Market Overview")

    try:

        # Calculate metrics for the selected date range
        comparison = calculate_period_comparison(
            filtered_prices
        )

        comparison = comparison[
            comparison["ticker"].isin(selected_tickers)
        ].copy()

        if comparison.empty:

            st.warning(
                "Not enough historical price data in this "
                "date range. Please select a wider period."
            )

        else:

            # Summary metrics
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Stocks Tracked",
                comparison["ticker"].nunique()
            )

            col2.metric(
                "Sectors Covered",
                comparison["sector"].nunique()
            )

            best_stock = comparison.iloc[0]

            col3.metric(
                "Top Historical Return",
                f'{best_stock["Cumulative_Return_Pct"]:.2f}%',
                help=f'Best performer: {best_stock["ticker"]}'
            )

            st.caption(
                f"Analysis period: {start_date:%d %b %Y} "
                f"to {end_date:%d %b %Y}"
            )

            st.divider()

            # Historical stock returns
            st.subheader("Stock Performance")

            top_stocks = comparison.sort_values(
                "Cumulative_Return_Pct",
                ascending=True
            )

            fig = px.bar(
                top_stocks,
                x="Cumulative_Return_Pct",
                y="ticker",
                orientation="h",
                color="Cumulative_Return_Pct",
                color_continuous_scale="RdYlGn",
                labels={
                    "ticker": "Stock",
                    "Cumulative_Return_Pct": (
                        "Cumulative Price Return (%)"
                    )
                },
                title="Historical Returns for Selected Stocks"
            )

            fig.update_yaxes(
                automargin=True
            )

            fig.update_layout(
                bargap=0.60,
                bargroupgap=0.25,
                height=500,
                margin=dict(
                    l=100,
                    r=60,
                    t=70,
                    b=60
                )
            )

            fig.update_layout(
                yaxis_title="",
                xaxis_title="Cumulative Price Return (%)",
                coloraxis_showscale=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                "Returns are measured from each stock's first available "
                "closing price to its latest available closing price "
                "within the selected period. They exclude dividends "
                "and do not predict future returns."
            )

            # Full comparison table
            st.subheader("Complete Stock Comparison")

            display_columns = [
                "Return_Rank",
                "ticker",
                "sector",
                "date",
                "close",
                "Cumulative_Return_Pct",
                "Annualized_Volatility_Pct",
                "Maximum_Drawdown_Pct"
            ]

            display_df = comparison[
                display_columns
            ].rename(
                columns={
                    "Return_Rank": "Rank",
                    "ticker": "Ticker",
                    "sector": "Sector",
                    "date": "Latest Price Date",
                    "close": "Latest Close",
                    "Cumulative_Return_Pct": (
                        "Cumulative Return (%)"
                    ),
                    "Annualized_Volatility_Pct": (
                        "Annualized Volatility (%)"
                    ),
                    "Maximum_Drawdown_Pct": (
                        "Maximum Drawdown (%)"
                    )
                }
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Latest Close": st.column_config.NumberColumn(
                        format="%.2f"
                    ),
                    "Cumulative Return (%)": (
                        st.column_config.NumberColumn(
                            format="%.2f%%"
                        )
                    ),
                    "Annualized Volatility (%)": (
                        st.column_config.NumberColumn(
                            format="%.2f%%"
                        )
                    ),
                    "Maximum Drawdown (%)": (
                        st.column_config.NumberColumn(
                            format="%.2f%%"
                        )
                    )
                }
            )

    except Exception as error:

        st.error(
            f"Could not load stock analytics: {error}"
        )

        st.info(
            "Check that PostgreSQL is running and your database "
            "connection settings are correct."
        )





elif page == "Stock Analysis":

    st.header("Stock Analysis")

    st.caption(
        "Explore historical price performance and risk "
        "within your selected analysis period."
    )

    try:
        # Recalculate metrics using the selected date range
        period_comparison = calculate_period_comparison(
            filtered_prices
        )

        if period_comparison.empty:

            st.warning(
                "Not enough historical price data in this "
                "date range. Please select a wider period."
            )

        else:


            selected_ticker = st.selectbox(
                "Select Stock",
                options=available_tickers,
                key="shared_single_stock"
            )


            # Filter historical prices for the selected stock
            stock_data = filtered_prices[
                filtered_prices["ticker"] == selected_ticker
            ].sort_values(
                "date"
            ).copy()

            # Get recalculated metrics for this stock
            metrics = period_comparison[
                period_comparison["ticker"] == selected_ticker
            ].iloc[0]

            st.caption(
                f"Analysis period: "
                f"{stock_data['date'].min():%d %b %Y} to "
                f"{stock_data['date'].max():%d %b %Y}"
            )

            # Display performance and risk metrics
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Cumulative Return",
                f'{metrics["Cumulative_Return_Pct"]:.2f}%'
            )

            col2.metric(
                "Annualized Volatility",
                f'{metrics["Annualized_Volatility_Pct"]:.2f}%'
            )

            col3.metric(
                "Maximum Drawdown",
                f'{metrics["Maximum_Drawdown_Pct"]:.2f}%'
            )

            st.divider()

            # Plot closing prices within the selected period
            st.subheader("Historical Price Movement")

            fig = px.line(
                stock_data,
                x="date",
                y="close",
                title=f"{selected_ticker} - Historical Closing Price",
                labels={
                    "date": "Date",
                    "close": "Closing Price (INR)"
                }
            )

            fig.update_layout(
                hovermode="x unified",
                xaxis_title="Date",
                yaxis_title="Closing Price (₹)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # Display the filtered historical data
            with st.expander("View Historical Price Data"):

                st.dataframe(
                    stock_data,
                    use_container_width=True,
                    hide_index=True
                )

            st.caption(
                "Returns are based on closing prices and exclude "
                "dividends. Volatility is annualized using 252 "
                "trading days. Historical metrics do not predict "
                "future performance."
            )

    except Exception as error:

        st.error(
            f"Unable to load stock analysis: {error}"
        )


elif page == "Technical Analysis":

    st.header("Technical Analysis")

    st.caption(
        "Explore candlestick price movements and exponential "
        "moving averages for individual stocks."
    )

    try:

        # Load historical OHLCV data from PostgreSQL
        import psycopg

        from src.data.db_connection import get_db_connection

        @st.cache_data(ttl=300)
        def load_technical_prices():

            query = """
                SELECT
                    date,
                    ticker,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM stock_prices
                ORDER BY date
            """

            connection = get_db_connection()

            try:

                with connection.cursor() as cursor:

                    cursor.execute(query)

                    rows = cursor.fetchall()

                    columns = [
                        description.name
                        for description in cursor.description
                    ]

                prices = pd.DataFrame(
                    rows,
                    columns=columns
                )

            finally:

                connection.close()

            prices["date"] = pd.to_datetime(
                prices["date"]
            )

            return prices

        technical_prices = load_technical_prices()

        # Select a stock
        available_tickers = sorted(
            technical_prices["ticker"].unique()
        )


        selected_ticker = st.selectbox(
            "Select Stock",
            options=available_tickers,
            key="shared_single_stock"
        )


        # Filter the selected stock
        stock_data = technical_prices[
            technical_prices["ticker"] == selected_ticker
        ].copy()

        stock_data = stock_data.sort_values(
            "date"
        )

        # Calculate moving averages using historical prices
        stock_data["EMA_20"] = (
            stock_data["close"].ewm(
                span=20,
                adjust=False
            ).mean()
        )

        stock_data["EMA_50"] = (
            stock_data["close"].ewm(
                span=50,
                adjust=False
            ).mean()
        )

        stock_data["EMA_200"] = (
            stock_data["close"].ewm(
                span=200,
                adjust=False
            ).mean()
        )

        
        # Calculate the 20-day average trading volume
        stock_data["Volume_MA_20"] = (
            stock_data["volume"]
            .rolling(window=20)
            .mean()
        )


        # Apply the selected analysis period after indicator calculation
        stock_data = stock_data[
            (stock_data["date"] >= start_date)
            & (stock_data["date"] <= end_date)
        ].copy()

        if stock_data.empty:

            st.warning(
                "No price data is available for this stock "
                "in the selected date range."
            )

        else:

            # Display latest price and moving averages
            latest = stock_data.iloc[-1]

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Latest Close",
                f'₹{latest["close"]:,.2f}'
            )

            col2.metric(
                "EMA 20",
                f'₹{latest["EMA_20"]:,.2f}'
            )

            col3.metric(
                "EMA 50",
                f'₹{latest["EMA_50"]:,.2f}'
            )

            col4.metric(
                "EMA 200",
                f'₹{latest["EMA_200"]:,.2f}'
            )

            st.divider()

            

            # Candlestick chart with a separate volume panel
            st.subheader(
                f"{selected_ticker} Price and Volume"
            )

            import plotly.graph_objects as go

            from plotly.subplots import make_subplots

            # Create two aligned chart panels
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.75, 0.25],
                subplot_titles=(
                    "Price and Moving Averages",
                    "Trading Volume"
                )
            )

            # Add candlesticks
            fig.add_trace(
                go.Candlestick(
                    x=stock_data["date"],
                    open=stock_data["open"],
                    high=stock_data["high"],
                    low=stock_data["low"],
                    close=stock_data["close"],
                    name="Price",
                    increasing_line_color="#26A69A",
                    decreasing_line_color="#EF5350",
                    increasing_fillcolor="#26A69A",
                    decreasing_fillcolor="#EF5350"
                ),
                row=1,
                col=1
            )

            # Add EMA 20
            fig.add_trace(
                go.Scatter(
                    x=stock_data["date"],
                    y=stock_data["EMA_20"],
                    mode="lines",
                    name="EMA 20",
                    line=dict(
                        color="#F5B041",
                        width=1.2
                    )
                ),
                row=1,
                col=1
            )

            # Add EMA 50
            fig.add_trace(
                go.Scatter(
                    x=stock_data["date"],
                    y=stock_data["EMA_50"],
                    mode="lines",
                    name="EMA 50",
                    line=dict(
                        color="#42A5F5",
                        width=1.2
                    )
                ),
                row=1,
                col=1
            )

            # Add EMA 200
            fig.add_trace(
                go.Scatter(
                    x=stock_data["date"],
                    y=stock_data["EMA_200"],
                    mode="lines",
                    name="EMA 200",
                    line=dict(
                        color="#AB7DF6",
                        width=1.5
                    )
                ),
                row=1,
                col=1
            )

            # Color volume bars according to daily price direction
            volume_colors = []

            for _, row in stock_data.iterrows():

                if row["close"] >= row["open"]:
                    volume_colors.append("#26A69A")

                else:
                    volume_colors.append("#EF5350")

            # Add daily volume bars
            fig.add_trace(
                go.Bar(
                    x=stock_data["date"],
                    y=stock_data["volume"],
                    name="Volume",
                    marker_color=volume_colors,
                    opacity=0.75
                ),
                row=2,
                col=1
            )

            # Add the 20-day average volume
            fig.add_trace(
                go.Scatter(
                    x=stock_data["date"],
                    y=stock_data["Volume_MA_20"],
                    mode="lines",
                    name="Volume MA 20",
                    line=dict(
                        color="#F5B041",
                        width=1.5
                    )
                ),
                row=2,
                col=1
            )

            # Style the chart
            fig.update_layout(
                template="plotly_dark",
                height=650,
                margin=dict(
                    l=15,
                    r=20,
                    t=75,
                    b=35
                ),
                hovermode="x unified",
                plot_bgcolor="#0B1020",
                paper_bgcolor="#0B1020",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.03,
                    xanchor="left",
                    x=0
                ),
                font=dict(
                    family="Arial, sans-serif",
                    size=12,
                    color="#D1D5DB"
                )
            )

            # Style the price panel
            # Set a tighter price-axis range
            price_columns = [
                "low",
                "high",
                "EMA_20",
                "EMA_50",
                "EMA_200"
            ]

            visible_min = stock_data[price_columns].min().min()
            visible_max = stock_data[price_columns].max().max()

            padding = (visible_max - visible_min) * 0.08

            fig.update_yaxes(
                title_text="Price (₹)",
                side="right",
                showgrid=True,
                gridcolor="rgba(156, 163, 175, 0.12)",
                zeroline=False,
                range=[
                    visible_min - padding,
                    visible_max + padding
                ],
                row=1,
                col=1
            )

            
            # Improve volume readability by limiting the effect
            # of unusually large volume spikes
            volume_upper = stock_data["volume"].quantile(0.99)

            fig.update_yaxes(
                title_text="Volume",
                side="right",
                showgrid=True,
                gridcolor="rgba(156, 163, 175, 0.12)",
                zeroline=False,
                range=[
                    0,
                    volume_upper * 1.10
                ],
                row=2,
                col=1
            )

            # Configure the upper price panel
            fig.update_xaxes(
                showgrid=False,
                showline=True,
                linecolor="#374151",
                tickformat="%b %Y",
                rangeslider_visible=False,
                row=1,
                col=1
            )

            # Keep the date labels visible on the lower volume panel
            fig.update_xaxes(
                showgrid=False,
                showline=True,
                linecolor="#374151",
                tickformat="%b %Y",
                rangeslider_visible=False,
                showticklabels=True,
                tickfont=dict(size=11, color="#D1D5DB"),
                row=2,
                col=1
            )
            

            # Focus on the most recent 126 trading sessions
            if len(stock_data) > 126:

                recent_start = stock_data["date"].iloc[-126]

                latest_date = stock_data["date"].iloc[-1]

                fig.update_xaxes(
                    range=[
                        recent_start,
                        latest_date
                    ],
                    row=1,
                    col=1
                )

                fig.update_xaxes(
                    range=[
                        recent_start,
                        latest_date
                    ],
                    row=2,
                    col=1
                )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                    "responsive": True
                }
            )


            st.caption(
                "EMA means Exponential Moving Average. Recent prices "
                "receive greater weight than older prices. These "
                "indicators describe historical price trends and "
                "are not standalone buy or sell signals."
            )

    except Exception as error:

        st.error(
            f"Unable to load technical analysis: {error}"
        )



elif page == "Sector Analysis":

    st.header("Sector Analysis")

    st.caption(
        "Compare historical performance and risk across five sectors."
    )

    try:

        # Calculate stock metrics for the selected date range
        stocks = calculate_period_comparison(
            filtered_prices
        )

        if stocks.empty:

            st.warning(
                "Not enough historical price data in this "
                "date range. Please select a wider period."
            )

        else:

            # Calculate sector averages from filtered stock metrics
            sectors = (
                stocks.groupby("sector")
                .agg(
                    Stock_Count=("ticker", "nunique"),
                    Average_Cumulative_Return_Pct=(
                        "Cumulative_Return_Pct",
                        "mean"
                    ),
                    Average_Annualized_Volatility_Pct=(
                        "Annualized_Volatility_Pct",
                        "mean"
                    ),
                    Average_Maximum_Drawdown_Pct=(
                        "Maximum_Drawdown_Pct",
                        "mean"
                    )
                )
                .reset_index()
            )

            # Summary metrics
            best_sector = sectors.loc[
                sectors["Average_Cumulative_Return_Pct"].idxmax()
            ]

            lowest_risk_sector = sectors.loc[
                sectors["Average_Annualized_Volatility_Pct"].idxmin()
            ]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Sectors Analysed",
                sectors["sector"].nunique()
            )

            col2.metric(
                "Top Performing Sector",
                best_sector["sector"],
                f'{best_sector["Average_Cumulative_Return_Pct"]:.2f}% avg. return'
            )

            col3.metric(
                "Lowest Volatility Sector",
                lowest_risk_sector["sector"],
                f'{lowest_risk_sector["Average_Annualized_Volatility_Pct"]:.2f}% annualized'
            )

            st.divider()

           # Sector return comparison
            st.subheader("Historical Sector Performance")

            sector_returns = sectors.sort_values(
                "Average_Cumulative_Return_Pct"
            ).copy()

            sector_returns["Return_Direction"] = sector_returns[
                "Average_Cumulative_Return_Pct"
            ].apply(
                lambda value: "Positive" if value >= 0 else "Negative"
            )

            fig = px.bar(
                sector_returns,
                x="Average_Cumulative_Return_Pct",
                y="sector",
                orientation="h",
                color="Return_Direction",
                color_discrete_map={
                    "Positive": "#22C55E",
                    "Negative": "#EF4444"
                },
                labels={
                    "sector": "Sector",
                    "Average_Cumulative_Return_Pct": (
                        "Average Price Return (%)"
                    ),
                    "Return_Direction": "Return"
                },
                title="Average Cumulative Price Return by Sector"
            )

            fig.update_yaxes(
                automargin=True
            )

            fig.update_layout(
                bargap=0.60,
                bargroupgap=0.25,
                height=500,
                margin=dict(
                    l=100,
                    r=60,
                    t=70,
                    b=60
                ),
                yaxis_title="",
                xaxis_title="Average Cumulative Price Return (%)",
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                "Sector figures are equal-weighted averages of the "
                "tracked stocks in each sector for the selected period. "
                "They are not actual sector-index or portfolio returns."
            )

            # Compare sector risk
            st.subheader("Sector Risk Comparison")

            risk_data = sectors.melt(
                id_vars="sector",
                value_vars=[
                    "Average_Annualized_Volatility_Pct",
                    "Average_Maximum_Drawdown_Pct"
                ],
                var_name="Risk_Metric",
                value_name="Percentage"
            )

            risk_data["Risk_Metric"] = risk_data[
                "Risk_Metric"
            ].map({
                "Average_Annualized_Volatility_Pct": (
                    "Annualized Volatility"
                ),
                "Average_Maximum_Drawdown_Pct": (
                    "Average Maximum Drawdown"
                )
            })

            fig2 = px.bar(
                risk_data,
                x="sector",
                y="Percentage",
                color="Risk_Metric",
                barmode="group",
                color_discrete_map={
                    "Annualized Volatility": "#60A5FA",
                    "Average Maximum Drawdown": "#EF4444"
                },
                labels={
                    "sector": "Sector",
                    "Percentage": "Percentage (%)",
                    "Risk_Metric": "Metric"
                },
                title="Average Volatility and Maximum Drawdown"
            )

            fig2.update_layout(
                bargap=0.60,
                bargroupgap=0.25,
                height=500,
                margin=dict(
                    l=100,
                    r=60,
                    t=70,
                    b=60
                )
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

            st.caption(
                "Drawdowns are negative. Values closer to zero indicate "
                "smaller average historical peak-to-trough declines."
            )

            # Inspect individual stocks within a sector
            st.subheader("Explore Stocks by Sector")

            selected_sector = st.selectbox(
                "Choose a sector",
                sorted(stocks["sector"].unique()),
                key="sector_stock_selector"
            )

            sector_stocks = stocks[
                stocks["sector"] == selected_sector
            ].sort_values(
                "Cumulative_Return_Pct",
                ascending=False
            )

            sector_display = sector_stocks[
                [
                    "ticker",
                    "Cumulative_Return_Pct",
                    "Annualized_Volatility_Pct",
                    "Maximum_Drawdown_Pct"
                ]
            ].rename(columns={
                "ticker": "Ticker",
                "Cumulative_Return_Pct": "Cumulative Return (%)",
                "Annualized_Volatility_Pct": "Annualized Volatility (%)",
                "Maximum_Drawdown_Pct": "Maximum Drawdown (%)"
            })

            st.dataframe(
                sector_display,
                column_config={
                    "Cumulative Return (%)": st.column_config.NumberColumn(
                        format="%.2f%%"
                    ),
                    "Annualized Volatility (%)": st.column_config.NumberColumn(
                        format="%.2f%%"
                    ),
                    "Maximum Drawdown (%)": st.column_config.NumberColumn(
                        format="%.2f%%"
                    )
                },
                use_container_width=True,
                hide_index=True
            )

            # Full sector summary
            with st.expander("View Complete Sector Summary"):

                sector_summary = sectors.sort_values(
                    "Average_Cumulative_Return_Pct",
                    ascending=False
                ).copy()

                st.dataframe(
                    sector_summary,
                    column_config={
                        "Average_Cumulative_Return_Pct": (
                            st.column_config.NumberColumn(
                                format="%.2f%%"
                            )
                        ),
                        "Average_Annualized_Volatility_Pct": (
                            st.column_config.NumberColumn(
                                format="%.2f%%"
                            )
                        ),
                        "Average_Maximum_Drawdown_Pct": (
                            st.column_config.NumberColumn(
                                format="%.2f%%"
                            )
                        )
                    },
                    use_container_width=True,
                    hide_index=True
                )

            st.caption(
                f"Metrics calculated for {start_date:%d %b %Y} "
                f"to {end_date:%d %b %Y}."
            )

    except Exception as error:

        st.error(
            f"Unable to load sector analysis: {error}"
        )

elif page == "Risk Analysis":

    st.header("Risk Analysis")

    st.caption(
        "Compare historical volatility, maximum drawdown, "
        "and returns across the tracked stocks."
    )

    try:

        # Calculate risk metrics for the selected period
        comparison = calculate_period_comparison(
            filtered_prices
        )

        comparison = comparison[
            comparison["ticker"].isin(selected_tickers)
        ].copy()

        if comparison.empty:

            st.warning(
                "Not enough historical price data in this "
                "date range. Please select a wider period."
            )

        else:

            # Identify the highest-risk and negative-return stocks
            highest_volatility = comparison.loc[
                comparison["Annualized_Volatility_Pct"].idxmax()
            ]

            smallest_drawdown = comparison.loc[
                comparison["Maximum_Drawdown_Pct"].idxmax()
            ]

            negative_returns = (
                comparison["Cumulative_Return_Pct"] < 0
            ).sum()

            # Display summary metrics
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Highest Volatility",
                highest_volatility["ticker"],
                f'{highest_volatility["Annualized_Volatility_Pct"]:.2f}%'
            )

            col2.metric(
                "Smallest Maximum Drawdown",
                smallest_drawdown["ticker"],
                f'{smallest_drawdown["Maximum_Drawdown_Pct"]:.2f}%'
            )

            col3.metric(
                "Stocks with Negative Returns",
                int(negative_returns),
                f"out of {len(comparison)} stocks"
            )

            st.divider()

            # Volatility comparison
            st.subheader("Annualized Volatility")

            fig_volatility = px.bar(
                comparison.sort_values(
                    "Annualized_Volatility_Pct"
                ),
                x="Annualized_Volatility_Pct",
                y="ticker",
                orientation="h",
                color="Annualized_Volatility_Pct",
                color_continuous_scale="OrRd",
                labels={
                    "ticker": "Stock",
                    "Annualized_Volatility_Pct": (
                        "Annualized Volatility (%)"
                    )
                },
                title="Historical Volatility by Stock"
            )

            fig_volatility.update_yaxes(
                automargin=True
            )

            fig_volatility.update_layout(
                bargap=0.60,
                bargroupgap=0.25,
                height=500,
                margin=dict(
                    l=100,
                    r=60,
                    t=70,
                    b=60
                )
            )

            fig_volatility.update_layout(
                height=450,
                template="plotly_dark",
                margin=dict(l=20, r=30, t=60, b=30),
                yaxis=dict(
                    title="",
                    automargin=True
                ),
                xaxis=dict(
                    title="Annualized Volatility (%)",
                    automargin=True
                )
            )

            st.plotly_chart(
                fig_volatility,
                use_container_width=True
            )

            # Maximum drawdown comparison
            st.subheader("Maximum Historical Drawdown")

            fig_drawdown = px.bar(
                comparison.sort_values(
                    "Maximum_Drawdown_Pct"
                ),
                x="Maximum_Drawdown_Pct",
                y="ticker",
                orientation="h",
                color="Maximum_Drawdown_Pct",
                color_continuous_scale="RdYlGn",
                labels={
                    "ticker": "Stock",
                    "Maximum_Drawdown_Pct": (
                        "Maximum Drawdown (%)"
                    )
                },
                title="Largest Historical Peak-to-Trough Declines"
            )

            fig_drawdown.update_layout(
                bargap=0.60,
                bargroupgap=0.25,
                height=500,
                margin=dict(
                    l=100,
                    r=60,
                    t=70,
                    b=60
                )
            )

            fig_drawdown.update_layout(
                height=450,
                template="plotly_dark",
                margin=dict(l=20, r=30, t=60, b=30),
                yaxis=dict(
                    title="",
                    automargin=True
                ),
                xaxis=dict(
                    title="Maximum Drawdown (%)",
                    automargin=True
                )
            )

            fig_drawdown.update_yaxes(
                automargin=True
            )

            st.plotly_chart(
                fig_drawdown,
                use_container_width=True
            )

            st.caption(
                "More negative drawdowns indicate deeper historical "
                "declines from a previous peak."
            )

            # Historical return versus volatility
            st.subheader("Historical Return vs Risk")

            fig_scatter = px.scatter(
                comparison,
                x="Annualized_Volatility_Pct",
                y="Cumulative_Return_Pct",
                color="sector",
                hover_name="ticker",
                labels={
                    "Annualized_Volatility_Pct": (
                        "Annualized Volatility (%)"
                    ),
                    "Cumulative_Return_Pct": (
                        "Cumulative Price Return (%)"
                    ),
                    "sector": "Sector"
                },
                title="Historical Return vs Annualized Volatility"
            )

            fig_scatter.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray"
            )

            st.plotly_chart(
                fig_scatter,
                use_container_width=True
            )

            st.caption(
                "Historical return and volatility describe past "
                "performance and risk; they do not predict future results."
            )

            # Individual stock risk profile
            st.subheader("Individual Stock Risk Profile")

            selected_ticker = st.selectbox(
                "Select a stock",
                sorted(comparison["ticker"].unique()),
                key="risk_stock_selector"
            )

            selected_stock = comparison[
                comparison["ticker"] == selected_ticker
            ].iloc[0]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Cumulative Return",
                f'{selected_stock["Cumulative_Return_Pct"]:.2f}%'
            )

            col2.metric(
                "Annualized Volatility",
                f'{selected_stock["Annualized_Volatility_Pct"]:.2f}%'
            )

            col3.metric(
                "Maximum Drawdown",
                f'{selected_stock["Maximum_Drawdown_Pct"]:.2f}%'
            )

            st.caption(
                f"Metrics calculated for the selected analysis period: "
                f"{start_date:%d %b %Y} to {end_date:%d %b %Y}."
            )

    except Exception as error:

        st.error(
            f"Unable to load risk analysis: {error}"
        )



elif page == "AI Stock Analyst":

    st.header("AI Stock Analyst")

    st.caption(
        "Understand historical stock performance and risk "
        "using transparent, data-driven explanations."
    )

    try:

        # Calculate metrics for the selected date range
        comparison = calculate_period_comparison(
            filtered_prices
        )

        if comparison.empty:

            st.warning(
                "Not enough historical price data in this "
                "date range. Please select a wider period."
            )

        else:

            selected_ticker = st.selectbox(
                "Select a stock to analyse",
                sorted(comparison["ticker"].unique()),
                key="analyst_stock_selector"
            )

            stock = comparison[
                comparison["ticker"] == selected_ticker
            ].iloc[0]

            # Exclude the selected stock from its sector peer group
            sector_peers = comparison[
                (comparison["sector"] == stock["sector"])
                & (comparison["ticker"] != selected_ticker)
            ]

            if sector_peers.empty:
                st.warning("No other tracked stocks are available for sector comparison.")
                st.stop()

            # Calculate equal-weighted peer averages
            sector_average_return = float(
                sector_peers["Cumulative_Return_Pct"].mean()
            )

            sector_average_volatility = float(
                sector_peers["Annualized_Volatility_Pct"].mean()
            )

            st.caption(
                f"Analysis period: {start_date:%d %b %Y} "
                f"to {end_date:%d %b %Y}"
            )

            st.divider()

            st.subheader(f"Analysis: {selected_ticker}")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Cumulative Return",
                f'{stock["Cumulative_Return_Pct"]:.2f}%'
            )

            col2.metric(
                "Annualized Volatility",
                f'{stock["Annualized_Volatility_Pct"]:.2f}%'
            )

            col3.metric(
                "Maximum Drawdown",
                f'{stock["Maximum_Drawdown_Pct"]:.2f}%'
            )

            st.divider()

            # Historical performance
            
            # Prepare structured metrics for Gemini

            stock_return = float(stock["Cumulative_Return_Pct"])

            volatility = float(stock["Annualized_Volatility_Pct"])

            drawdown = float(stock["Maximum_Drawdown_Pct"])

            stock_metrics = {
                "ticker": selected_ticker,
                "sector": stock["sector"],
                "analysis_period": (
                    f"{start_date:%d %b %Y} to "
                    f"{end_date:%d %b %Y}"
                ),
                "cumulative_return_pct": round(
                    float(stock_return), 2
                ),
                "annualized_volatility_pct": round(
                    float(volatility), 2
                ),
                "maximum_drawdown_pct": round(
                    float(drawdown), 2
                ),
                "sector_average_return_pct": round(
                    float(sector_average_return), 2
                ),
                "sector_average_volatility_pct": round(
                    float(sector_average_volatility), 2
                ),
                "sector_peer_count": len(sector_peers)
            }

            st.divider()

            st.subheader("AI-Generated Stock Analysis")

            st.caption(
                "Gemini interprets historical metrics calculated "
                "from the selected stock and its sector peers."
            )

            # Generate analysis only when requested
            if st.button(
                "Generate AI Analysis",
                type="primary",
                key="generate_stock_analysis"
            ):

                with st.spinner(
                    "Gemini is analysing the historical stock data..."
                ):

                    try:

                        analysis = cached_stock_analysis(
                            stock_metrics
                        )

                        st.session_state["analyst_result"] = {
                            "ticker": selected_ticker,
                            "period": (
                                f"{start_date:%d %b %Y} to "
                                f"{end_date:%d %b %Y}"
                            ),
                            "content": analysis
                        }

                    except Exception as error:

                        st.error(
                            f"AI analysis failed: {error}"
                        )

            # Display analysis for the current stock and period
            result = st.session_state.get("analyst_result")

            current_period = (
                f"{start_date:%d %b %Y} to "
                f"{end_date:%d %b %Y}"
            )

            if (
                result
                and result["ticker"] == selected_ticker
                and result["period"] == current_period
            ):

                st.markdown(result["content"])

            else:

                st.info(
                    "Click Generate AI Analysis to generate "
                    "an explanation for the selected stock."
                )

            st.divider()

            st.caption(
                "Analysis is based exclusively on historical metrics "
                "and sector comparisons. It does not predict future "
                "returns or provide personalised investment advice."
            )


    except Exception as error:

        st.error(
            f"Unable to generate stock analysis: {error}"
        )
