# 🏦 Market Maker & Execution Strategy Simulator (Streamlit App)

This project is a Streamlit-based simulation of two core trading roles in electronic markets:

1. **Market Making** – continuously quoting bid/ask prices and managing inventory
2. **Execution Strategy** – placing limit orders to fill a target position with minimal market impact

It’s designed for educational and prototyping purposes in quant trading, strategy development, and microstructure research.

---

## 🎯 Key Features

### 📈 Market Making Module
- Adaptive bid/ask pricing based on inventory levels
- Randomized market order flow to simulate fill risk
- Inventory and P&L tracking
- Price/time visualizations

### 🎯 Execution Strategy Module
- Limit order execution to achieve a target position
- Queue modeling and partial fills
- Customizable fill probabilities
- Visual fill tracking over time

---

## 🧰 Tech Stack

- Python
- Streamlit
- NumPy
- Pandas
- Matplotlib or Plotly

---

## 🖥️ Run the App Locally

```bash
git clone https://github.com/Ftariq17/market_maker_sim.git
cd market_maker_sim
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
streamlit run main.py
