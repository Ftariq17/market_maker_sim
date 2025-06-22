
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.title("Trading Simulator")

# --- Mode Selection ---
simulation_mode = st.sidebar.radio("Simulation Mode", ["Market Making", "Execution Strategy"])

# ------------------------------
# Market Making Simulator Logic
# ------------------------------
def run_market_making_simulation():
    st.subheader("Market Making Simulator")
    quoting_mode = st.sidebar.selectbox("Quoting Mode", ["adaptive", "passive", "aggressive"])
    T = st.sidebar.slider("Simulation Steps", 100, 20000, 5000, step=100)
    volatility = st.sidebar.slider("Volatility (%)", 0.1, 5.0, 1.0) / 100
    base_spread_pct = st.sidebar.slider("Base Spread (%)", 0.01, 1.0, 0.2) / 100
    inventory_skew_factor = st.sidebar.slider("Inventory Skew Factor", 0.001, 0.05, 0.01)
    inventory_penalty_strength = st.sidebar.slider("Inventory Penalty Strength", 0.01, 1.0, 0.3)
    max_inventory = st.sidebar.slider("Max Inventory", 1, 20, 6)
    latency_steps = st.sidebar.slider("Latency (steps)", 0, 10, 2)
    shock_probability = st.sidebar.slider("Shock Frequency (%)", 0.0, 10.0, 1.0) / 100
    shock_magnitude = st.sidebar.slider("Shock Size (%)", 0.1, 10.0, 5.0) / 100

    np.random.seed(42)
    midprices = [100.0]
    bids, asks = [], []
    inventory, cash = [0], [0.0]
    pnl_total, pnl_spread, pnl_inventory = [], [], []
    quote_buffer = []
    spread_history = []
    trade_log = []

    inv, c, spread_pnl, inv_pnl = 0, 0.0, 0.0, 0.0

    for t in range(T):
        prev_mid = midprices[-1]
        step_return = np.random.normal(0, volatility)
        if np.random.rand() < shock_probability:
            step_return += np.random.choice([-1, 1]) * shock_magnitude
        mid = prev_mid * (1 + step_return)
        midprices.append(mid)

        recent_window = midprices[-min(50, len(midprices)):]
        realized_vol = np.std(recent_window) if len(recent_window) > 1 else volatility
        spread = base_spread_pct * mid * (1 + 5 * realized_vol)
        spread_history.append(spread)

        skew = inventory_skew_factor * inv
        raw_bid = mid - spread / 2 - skew
        raw_ask = mid + spread / 2 - skew

        if quoting_mode == "passive":
            bid, ask = raw_bid - 0.01, raw_ask + 0.01
        elif quoting_mode == "aggressive":
            bid, ask = raw_bid + 0.01, raw_ask - 0.01
        else:
            adj = 0.02 * np.sign(-inv) if abs(inv) > max_inventory * 0.5 else 0
            bid, ask = raw_bid + adj, raw_ask + adj

        quote_buffer.append((bid, ask))
        if len(quote_buffer) <= latency_steps:
            bid, ask = quote_buffer[0]
        else:
            bid, ask = quote_buffer[-latency_steps]

        bids.append(bid)
        asks.append(ask)

        fill_prob_bid = max(0, 1 - abs(mid - bid) / spread)
        fill_prob_ask = max(0, 1 - abs(mid - ask) / spread)
        fill_bid = np.random.rand() < fill_prob_bid and inv < max_inventory
        fill_ask = np.random.rand() < fill_prob_ask and inv > -max_inventory

        if fill_bid:
            inv += 1
            c -= bid
            trade_log.append({'Step': t, 'Side': 'Buy', 'Price': bid, 'Inventory': inv})

        if fill_ask:
            inv -= 1
            c += ask
            trade_log.append({'Step': t, 'Side': 'Sell', 'Price': ask, 'Inventory': inv})

        if fill_bid and fill_ask:
            spread_pnl += ask - bid

        inv_pnl += inv * (mid - prev_mid)
        inv_penalty = inventory_penalty_strength * (inv ** 2)
        inventory.append(inv)
        cash.append(c)
        pnl_spread.append(spread_pnl)
        pnl_inventory.append(inv_pnl)
        pnl_total.append(c + inv * mid - inv_penalty)

    df = pd.DataFrame({
        'Midprice': midprices[1:],
        'Bid': bids,
        'Ask': asks,
        'Inventory': inventory[1:],
        'Cash': cash[1:],
        'PnL_Total': pnl_total,
        'PnL_Spread': pnl_spread,
        'PnL_Inventory': pnl_inventory,
        'Spread': spread_history,
    })

    num_bid_fills = sum(1 for t in trade_log if t['Side'] == 'Buy')
    num_ask_fills = sum(1 for t in trade_log if t['Side'] == 'Sell')

    st.subheader("Midprice, Bid, and Ask")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(y=df['Midprice'], mode='lines', name='Midprice', line=dict(color='black', width=3)))
    fig1.add_trace(go.Scatter(y=df['Bid'], mode='lines', name=f'Bid (Fills: {num_bid_fills})', line=dict(color='skyblue', dash='dot')))
    fig1.add_trace(go.Scatter(y=df['Ask'], mode='lines', name=f'Ask (Fills: {num_ask_fills})', line=dict(color='orange', dash='dash')))
    fig1.update_layout(height=500, xaxis_title='Step', yaxis_title='Price')
    st.plotly_chart(fig1)

    st.subheader("PnL Over Time")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=df['PnL_Total'], name='Total PnL', line=dict(color='green', width=4)))
    fig2.add_trace(go.Scatter(y=df['PnL_Spread'], name='Spread PnL', line=dict(color='blue', width=3, dash='dash')))
    fig2.add_trace(go.Scatter(y=df['PnL_Inventory'], name='Inventory PnL', line=dict(color='skyblue', width=3, dash='dot')))
    fig2.update_layout(height=500, xaxis_title='Step', yaxis_title='PnL')
    st.plotly_chart(fig2)

    st.subheader("Inventory Distribution")
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=df['Inventory'], nbinsx=20, name='Inventory', marker=dict(color='rgba(135, 206, 235, 0.6)', line=dict(width=1.5, color='black'))))
    fig3.update_layout(height=300, xaxis_title='Inventory Level', yaxis_title='Frequency')
    st.plotly_chart(fig3)

    st.subheader("Final Metrics")
    col1, col2, col3, col4 = st.columns(4)
    def render_card(label, value, is_inventory=False):
        if is_inventory:
            border_color = 'rgba(0, 128, 0, 0.6)' if value == 0 else 'rgba(255, 165, 0, 0.6)'
            text_color = 'rgba(0, 128, 0, 1)' if value == 0 else 'rgba(255, 140, 0, 1)'
        else:
            border_color = 'rgba(0, 128, 0, 0.6)' if value >= 0 else 'rgba(255, 165, 0, 0.6)'
            text_color = 'rgba(0, 128, 0, 1)' if value >= 0 else 'rgba(255, 140, 0, 1)'
        return f"""
            <div style='padding: 1em; background-color: #f9f9f9; border-left: 5px solid {border_color}; border-radius: 8px; text-align: center; box-shadow: 1px 1px 3px rgba(0,0,0,0.1);'>
                <div style='font-size: 0.9em; color: #444;'>{label}</div>
                <div style='font-size: 1.5em; font-weight: bold; color: {text_color};'>{value:.2f}</div>
            </div>
        """

    with col1:
        st.markdown(render_card("Final PnL", pnl_total[-1]), unsafe_allow_html=True)
    with col2:
        st.markdown(render_card("Final Inventory", inventory[-1], is_inventory=True), unsafe_allow_html=True)
    with col3:
        st.markdown(render_card("Total Spread PnL", pnl_spread[-1]), unsafe_allow_html=True)
    with col4:
        st.markdown(render_card("Total Inventory PnL", pnl_inventory[-1]), unsafe_allow_html=True)

    st.subheader("Trade Log")
    log_df = pd.DataFrame(trade_log)
    st.dataframe(log_df)
    st.download_button("Download Trade Log as CSV", data=log_df.to_csv(index=False).encode('utf-8'), file_name='trade_log.csv', mime='text/csv')

# ------------------------------
# Execution Strategy Simulator
# ------------------------------
def run_execution_strategy_simulator():
    st.subheader("Execution Strategy Simulator (Full Order Book)")
    T = st.sidebar.slider("Execution Steps", 100, 20000, 5000, step=100)
    latency_steps = st.sidebar.slider("Latency (steps)", 0, 20, 5)
    order_price_offset = st.sidebar.slider("Limit Order Offset from Mid (%)", 0.0, 1.0, 0.05) / 100
    order_qty = st.sidebar.slider("Order Quantity", 1, 50, 5)
    timeout_steps = st.sidebar.slider("Order Timeout (steps)", 10, 200, 50)

    np.random.seed(42)
    midprices = [100.0]
    volatility = 0.002

    executed_trades = []
    order_log = []
    your_orders = []

    book_levels = 3
    book_tick = 0.05

    for t in range(T):
        prev_mid = midprices[-1]
        step_return = np.random.normal(0, volatility)
        mid = prev_mid * (1 + step_return)
        midprices.append(mid)

        mid_rounded = round(mid / book_tick) * book_tick

        bid_prices = [round(mid_rounded - i * book_tick, 2) for i in range(1, book_levels + 1)]
        ask_prices = [round(mid_rounded + i * book_tick, 2) for i in range(1, book_levels + 1)]

        order_book = {
            'bid': {p: np.random.randint(80, 150) for p in bid_prices},
            'ask': {p: np.random.randint(80, 150) for p in ask_prices}
        }

        # Submit an order every 200 steps at best ask
        if t % 200 == 0:
            if order_book['ask']:
                target_price = sorted(order_book['ask'].keys())[0]  # Best ask price
                queue_at_price = order_book['ask'].get(target_price, 100)
                order = {
                    'submit_time': t,
                    'exec_time': t + latency_steps,
                    'price': target_price,
                    'qty': order_qty,
                    'queue_pos': queue_at_price,
                    'status': 'pending'
                }
                your_orders.append(order)
                order_log.append(order)

        market_sell_volume = np.random.poisson(80)

        for price in sorted(order_book['ask'].keys()):
            volume_at_price = order_book['ask'][price]
            if market_sell_volume <= 0:
                break
            trade_size = min(market_sell_volume, volume_at_price)
            order_book['ask'][price] -= trade_size
            market_sell_volume -= trade_size

            for order in your_orders:
                if order['status'] == 'pending' and order['exec_time'] <= t and order['price'] == price:
                    if trade_size >= order['queue_pos']:
                        fill_amount = order['qty']
                        order['status'] = 'filled'
                        executed_trades.append({
                            'step': t,
                            'price': price,
                            'qty': fill_amount,
                            'mid_at_fill': mid
                        })
                        trade_size -= order['queue_pos']
                        order['queue_pos'] = 0
                    else:
                        order['queue_pos'] -= trade_size
                        trade_size = 0

        for o in your_orders:
            if o['status'] == 'pending' and t - o['submit_time'] > timeout_steps:
                o['status'] = 'expired'

    exec_df = pd.DataFrame(executed_trades)
    if not exec_df.empty:
        exec_df['slippage'] = exec_df['mid_at_fill'] - exec_df['price']
        total_filled = exec_df['qty'].sum()
        avg_slippage = exec_df['slippage'].mean()
    else:
        total_filled = 0
        avg_slippage = 0.0
    total_orders = len(order_log)
    fill_rate = total_filled / (total_orders * order_qty) if total_orders > 0 else 0

    st.subheader("Midprice Over Time")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(y=midprices, mode='lines', name='Midprice'))
    st.plotly_chart(fig1)

    st.subheader("Filled Trades")
    if not exec_df.empty:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=exec_df['step'], y=exec_df['price'], mode='markers', name='Fill Price'))
        fig2.add_trace(go.Scatter(x=exec_df['step'], y=exec_df['mid_at_fill'], mode='lines', name='Midprice at Fill'))
        st.plotly_chart(fig2)
    else:
        st.info("No trades were filled during the simulation.")

    st.subheader("Execution Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Filled Quantity", f"{total_filled}")
    col2.metric("Average Slippage", f"{avg_slippage:.4f}")
    col3.metric("Fill Rate", f"{fill_rate:.2%}")

    st.subheader("Order Log")
    log_df = pd.DataFrame(order_log)
    st.dataframe(log_df)
    if not log_df.empty:
        st.download_button("Download Order Log as CSV", data=log_df.to_csv(index=False).encode('utf-8'), file_name='execution_order_log.csv', mime='text/csv')

# ------------------------------
# Run Selected Simulation
# ------------------------------
if simulation_mode == "Market Making":
    run_market_making_simulation()
else:
    run_execution_strategy_simulator()
