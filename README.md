# Financial Renewable Power Purchasing Agreement (PPA) - Valuation using a Stochastic Pricing Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://renewable-ppa-pricing-engine.streamlit.app/)

> **⚠️ Disclaimer**
> This repository is an independent project entirely built outside of professional hours for educational purposes. It does not constitute financial or trading advice. All underlying data used in this model (including EEX futures curves and ECB yield curves) is either synthetic or publicly available. This codebase does not contain, reflect, or make use of any proprietary data, models, risk frameworks, or intellectual property from any current or former employer.

## 📌 Commercial Overview
This interactive pricing engine is designed to model the exact mark-to-market (MtM) and Potential Future Exposure (PFE) profiles of a financial renewable Power Purchase Agreement (PPA) from the perspective of an originator operating in the **DK1 (Western Denmark)** [bidding zone(https://www.acer.europa.eu/electricity/market-rules/capacity-allocation-and-congestion-management/bidding-zone-review). The user can choose from three distinct assets: Solar, Wind, and BESS.

For operators in European grids, extreme grid tail-risks (e.g., severe supply gluts or weather-driven price crashes) and renewable cannibalization are important sources of risk. This engine allows the user to dynamically test the fixed leg of a flex structure that accommodates these dynamics via a live Streamlit dashboard.

## 📐 Quantitative Architecture
The stochastic engine driving the simulations rests on a **Mean-Reverting, Mixed Jump-Diffusion Model**, thereby accounting for the price spikes characteristic of European power grids.

1. **Diffusion Process (Augmented Ornstein-Uhlenbeck Process):** 
   Energy prices have been found to exhibit mean-reversion, as they are pulled toward a long-term average price. The Augmented Ornstein-Uhlenbeck Process models the baseline price variance, using EEX futures contracts prices as the dynamic "center of gravity". The model's parameters (baseline price variance and mean-reversion speed, or strength) are obtained via Maximum Likelihood Estimation (MLE).
2. **Jump Component (Peak-Over-Threshold):** 
   Extreme Value Theory is used to isolate tail events. A Poisson process generates discrete market shocks, accounted for using exact, closed-form analytical expressions.
3. **Cannibalization Beta & Gamma:** 
   Account for the (non-linear) relationship between renewable generation and grid saturation. In effect, the originator can be reasonably expected to capture only a fraction of the baseload price during periods of high renewable energy generation.

## ⚙️ Asset-Specific Generation Profiles
The engine applies specific grid constraints and capture logic to each asset class:

* **☀️ Solar (Deterministic Capture Discount):** Solar generation is heavily concentrated during midday hours, aligning with the well-known "duck curve" and pushing daytime prices down. The model applies a static capture discount to the stochastic baseload price, reflecting the asset's inability to participate in lucrative evening peaks.
* **🌪️ Wind (Volume Auto-Correlation):** Wind generation is stochastic but heavily auto-correlated across the grid. The model employs a polynomial regression to dynamically adjust the effective capacity factor against the simulated daily price. High-volume days mathematically trigger the non-linear acceleration of market cannibalization.
* **🔋 BESS (Intraday Spot Arbitrage):** A Battery Energy Storage System does not utilize a fixed PPA leg. It operates as a merchant asset, moving power through time. The engine bypasses the fixed-leg logic and simulates intraday spread harvesting, assuming the asset charges during the lowest daily percentiles and discharges during the highest, explicitly netting out Round-Trip Efficiency (RTE) thermal losses.

## 🚀 Live Interactive Dashboard
The quantitative model is wrapped in a front-office commercial UI. The user can run real-time Monte Carlo simulations and test pricing strategies without configuring a local Python environment.

👉 **[Access the Live Streamlit App Here](https://renewable-ppa-pricing-engine.streamlit.app/)**

## 💻 Local Installation & Execution
If the user wishes to review the underlying `app.py` codebase or run the Monte Carlo engine locally, they can clone this repository:
   ```bash
   git clone [https://github.com/g-franchini/renewable-ppa-pricing-engine.git](https://github.com/g-franchini/renewable-ppa-pricing-engine.git)
   cd renewable-ppa-pricing-engine
   pip install -r requirements.txt
   streamlit run app.py
   ```

Thank you for reviewing this project. Any feedback is highly appreciated!
