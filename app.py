import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import statsmodels.api as sm
import plotly.graph_objects as go

np.random.seed(42)

# ---------------------------------------------------------
# 1. PAGE CONFIG & STYLING (Front-Office Aesthetic)
# ---------------------------------------------------------
st.set_page_config(page_title="PPA Flex Structuring Sandbox", layout="wide")
st.title("⚡ Renewable PPA Flex Valuation & Extreme Tail-Risk Sandbox")
st.markdown("---")

# Initialize persistent session state for tracking user experiments
if 'experiment_history' not in st.session_state:
    st.session_state.experiment_history = []

# =====================================================================
# ABSOLUTE MEAN-REVERTING JUMP-DIFFUSION MODEL
# (AUGMENTED ORNSTEIN-UHLENBECK PROCESS)
# =====================================================================

# ---------------------------------------------------------
# CANNIBALIZATION & DATA LOADING
# ---------------------------------------------------------
@st.cache_data
def load_historical_data():
    """Loads CSV once into memory to prevent UI lag on slider movement."""
    return pd.read_csv('./data/PRICES_AND_GENERATION_20231231_20260531.csv')

@st.cache_data
def calibrate_cannibalization(df_daily, generation_col='Generation'):
    """Regresses Z-Scored Prices and Squared Z-Scored Prices against Capacity Factor.
    This is meant to capture the (non-linear) effect of cannibalization. """
    
    # Normalize the generation to a Capacity Factor (0.0 to 1.0)
    max_gen = df_daily[generation_col].max()
    df_daily['Renewable_CF'] = df_daily[generation_col] / max_gen
    
    price_mean = df_daily['Price'].mean()
    price_std = df_daily['Price'].std()

    df_daily['Price_Z'] = (df_daily['Price'] - price_mean) / price_std
    # Squared term for the non-linear tail acceleration
    df_daily['Price_Z_Sq'] = df_daily['Price_Z'] ** 2

    X = sm.add_constant(df_daily[['Price_Z', 'Price_Z_Sq']]) 
    Y = df_daily['Renewable_CF']

    # OLS
    ols_model = sm.OLS(Y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 7})
    
    alpha = ols_model.params['const']
    beta = ols_model.params['Price_Z']
    gamma = ols_model.params['Price_Z_Sq']

    # Extract significance metrics
    r_squared = ols_model.rsquared
    p_values = ols_model.pvalues
    
    # Locally check the fit
    print(f"Cannibalization R-Squared: {r_squared:.4f}")
    print(f"P-Values:\n{p_values}")
    
    return alpha, beta, gamma, price_mean, price_std

df_agg = load_historical_data()
alpha_emp, beta_emp, gamma_emp, p_mean, p_std = calibrate_cannibalization(df_agg)

# =====================================================================
# CALIBRATION ENGINE
# =====================================================================
class StochasticCalibrator:
    def __init__(self, daily_df):
        self.prices = daily_df['Price'].values
        # Assuming 365 days (and corresponding discrete-time time step)
        self.dt = 1.0 / 365.0
        
    def calibrate_jumps(self, threshold_std=3.0):
        """Peak-Over-Threshold (POT) to extract Poisson parameters using exact, closed-form analytical expressions."""
        # Absolute Changes in the Price of the Asset
        dS = self.prices[1:] - self.prices[:-1]
        # Standard Deviation of the Absolute Changes in the Price of the Asset
        std_dS = np.std(dS)
        
        # Identify days characterized by the Poisson Process
        jumps = dS[np.abs(dS) > threshold_std * std_dS]

        # Jump Frequency (Hazard Rate) - How many jumps happen in a year
        self.lambda_j = (len(jumps) / len(self.prices)) * 365.0
        # Average Size of a Jump; how big is the spike, on average?
        self.mu_j = np.mean(jumps) if len(jumps) > 0 else 0.0
         # Standard Deviation of the Jump
        self.sigma_j = np.std(jumps) if len(jumps) > 0 else 0.0
        
        # Identify days characterized by the Diffusion Process
        self.diffusion_filter = np.abs(dS) <= threshold_std * std_dS
        self.normal_S_prev = self.prices[:-1][self.diffusion_filter]
        self.normal_S_curr = self.prices[1:][self.diffusion_filter]
        
        return self.lambda_j, self.mu_j, self.sigma_j
    
    def calibrate_diffusion(self):
        """Maximum Likelihood Estimation (MLE) for the parameters of the Ornstein-Uhlenbeck diffusion process."""
        N = len(self.normal_S_curr)
        hist_mean = np.mean(self.normal_S_curr)

        def negative_log_likelihood(params):
            kappa, sigma = params
            if kappa <= 0 or sigma <= 0: return 1e10

            # Exact Conditional Distribution is available for the Ornstein-Uhlenbeck diffusion process,
            # allowing us to bypass the (error-prone) Euler approximation for discrete-time calculation
            # Conditional Mean & Variance
            mu = self.normal_S_prev * np.exp(-kappa * self.dt) + hist_mean * (1 - np.exp(-kappa * self.dt))
            var = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * self.dt))
            
            # Log-Likelihood equation (retrieving the parameters that make past observations as likely to have happened as possible)
            LL = - (N / 2.0) * np.log(2 * np.pi * var) - np.sum((self.normal_S_curr - mu)**2) / (2 * var)
            return -LL

        # Initial guesses and bounds
        initial_guess = [5.0, np.std(self.normal_S_curr)]
        bounds = ((1e-5, None), (1e-5, None))

        result = minimize(negative_log_likelihood, initial_guess, bounds=bounds, method='L-BFGS-B')
        self.kappa, self.sigma = result.x
        
        return self.kappa, self.sigma
    
# =====================================================================
# SIDEBAR CONTROL PANEL (The Commercial Levers)
# =====================================================================
st.sidebar.header("🕹️ Commercial & Risk Levers")

st.sidebar.subheader("1. Commercial Structure")

asset_class = st.sidebar.selectbox(
    "Asset Profile / Technology", 
    ["Solar ('Deterministic' Capture Discount)", "Wind (Volume Correlation)", "BESS (Intraday Arbitrage)"]
)

# Dynamically hide the Fixed Leg slider if BESS is selected
if "BESS" in asset_class:
    st.sidebar.info("🔋 BESS operates as a merchant asset (Spot Arbitrage). Fixed PPA leg is not applicable.")
    fixed_leg_input = 0.0  # Dummy value to pass into the model engine
else:
    fixed_leg_input = st.sidebar.slider("PPA Strike Price (Fixed Leg, €/MWh)", min_value=40.0, max_value=120.0, value=82.0, step=1.0)

st.sidebar.subheader("2. Tail-Risk Component (POT)")
st.sidebar.markdown("Adjusting the threshold dynamically alters the baseline diffusion MLE calibration.")
threshold_input = st.sidebar.slider("Jump Threshold (Standard Deviations)", min_value=1.5, max_value=6.0, value=3.0, step=0.25)

calibrator = StochasticCalibrator(df_agg)
lambda_j, mu_j, sigma_j = calibrator.calibrate_jumps(threshold_std=threshold_input)
kappa_base, sigma_base = calibrator.calibrate_diffusion()

st.sidebar.subheader("3. Diffusion Volatility")
st.sidebar.markdown(f"MLE Calibrated Baseline: **{sigma_base:.2f}**")
sigma_input = st.sidebar.slider(
    "Diffusion Volatility (σ)", 
    min_value=float(sigma_base * 0.1), 
    max_value=float(sigma_base * 3.0), 
    value=float(sigma_base), 
    step=1.0
)

st.sidebar.info(
    f"**Live MLE Output:**\n"
    f"* Jump Frequency - Hazard Rate (λ): {lambda_j:.2f}/yr\n"
    f"* Jump Mean (μ): €{mu_j:.2f}\n"
    f"* Jump Standard Deviation (σ): €{sigma_j:.2f}\n"
    f"* Mean-Reversion (κ): {kappa_base:.3f}"
)

# =====================================================================
# MARKET CURVES & MONTE CARLO INITIALIZATION
# =====================================================================
S0 = df_agg['Price'].iloc[-1]
timeline_dates = pd.date_range(start="2026-06-01", periods=int(5 * 365 + 1), freq="D")

# EEX Futures Contracts (consensus of long-term average price; "center of gravity")
contract_map = {
        "2026-06-01": 105.25, 
        "2026-07-01": 101.85, # Front Month
        "2026-08-01": 100.96, # Front Month
        "2026-09-01": 105.71, # Front Month
        "2026-10-01": 99.92, # Front Month
        "2026-11-01": 113.81, # Front Month
        "2026-12-01": (3 * 109.02) - 113.81 - 99.92, # Front Month (data unavailable in EEX; using "dirty" approximation)
        "2027-01-01": 100.53, # 2027 Q1
        "2027-04-01": 77.07, # 2027 Q2
        "2027-07-01": 80.78, # 2027 Q3
        "2027-10-01": 86.24, # 2027 Q4
        "2028-01-01": 75.96, # Cal 28
        "2029-01-01": 72.42, # Cal 29
        "2030-01-01": 68.80, # Cal 30
        "2031-01-01": 70.59 # Cal 31
}

fwd_df = pd.DataFrame(index=timeline_dates)
fwd_df['Price'] = np.nan
for date_str, price in contract_map.items():
    if pd.to_datetime(date_str) in fwd_df.index:
        fwd_df.loc[date_str, 'Price'] = price
fwd_prices = fwd_df['Price'].ffill().values

# ECB Data Portal
# Tenors: 0.5, 1, 1.5, 2, 3, 4, 5
yield_rates = np.array([0.020, 0.0258, 0.0280, 0.0287, 0.0288, 0.0289, 0.0291])

# =====================================================================
# MONTE CARLO
# =====================================================================
class PPA_Exposure_Model:
    def __init__(self, S0, kappa, sigma, lambda_j, mu_j, sigma_j, 
                 fwd_prices, yield_rates, asset,
                 alpha_vol, beta_vol, gamma_vol, price_mean, price_std,
                 max_capacity=48.0, fixed_price=82.0, paths=1000):
        
        # Asset Class
        self.asset_class = asset
        
        # Initial Price of the Asset, Mean-Reversion Speed, Volality of the Diffusion Process
        self.S0, self.kappa, self.sigma = S0, kappa, sigma
        # Hazard Rate (# jumps), Average Jump Size, Standard Deviation of the Jump Size
        self.lambda_j, self.mu_j, self.sigma_j = lambda_j, mu_j, sigma_j
        
        # Time variables (defined with respect to the number of forward prices)
        self.trading_days = len(fwd_prices) 
        self.dt = 1.0 / 365.0
        self.years = self.trading_days / 365.0

        self.fwd_days_array = np.arange(self.trading_days)
        self.time_array_yrs = np.linspace(0, self.years, self.trading_days) 
        self.paths = paths 

        # Price of the fixed leg of the fixed-for-float PPA
        self.fixed_price = fixed_price
        # Maximum agreed-upon capacity (MW)
        # "Effective" capacity can change subject to the effect of cannibalization
        self.max_capacity = max_capacity
        
        # OLS Parameters
        self.alpha_vol, self.beta_vol, self.gamma_vol = alpha_vol, beta_vol, gamma_vol
        self.price_mean, self.price_std = price_mean, price_std
        
        # Interpolators (w/ Cubic Spline for the Yield Curve)
        self.forward_curve = interp1d(self.fwd_days_array, fwd_prices, kind='linear', fill_value='extrapolate')
        yield_tenors = np.linspace(0, 5.0, len(yield_rates))
        self.yield_curve = interp1d(yield_tenors, yield_rates, kind='cubic', fill_value='extrapolate')
        
        daily_rates = self.yield_curve(self.time_array_yrs)
        self.discount_factor = np.exp(-daily_rates * self.time_array_yrs)
        
    def simulate_prices(self):
        """Simulates paths using the Forward Curve as the dynamic center of gravity."""
        self.simulated_prices = np.zeros((self.paths, self.trading_days))
        self.simulated_prices[:, 0] = self.S0
        
        # Exact Conditional Distribution is available for the Ornstein-Uhlenbeck diffusion process,
        # allowing us to bypass the (error-prone) Euler approximation for discrete-time calculation
        # Conditional Variance
        exact_var = (self.sigma**2 / (2 * self.kappa)) * (1 - np.exp(-2 * self.kappa * self.dt))
        exact_std = np.sqrt(exact_var)
        
        # "Random" noise and (normally distributed) jumps
        Z = np.random.standard_normal((self.paths, self.trading_days))
        jump_flags = np.random.poisson(self.lambda_j * self.dt, (self.paths, self.trading_days))
        jump_sizes = np.random.normal(self.mu_j, self.sigma_j, (self.paths, self.trading_days))
        
        for t in range(1, self.trading_days):
            # Previous day's average price
            S_prev = self.simulated_prices[:, t-1]

            # Forward curve to get the long-term average price of electricity at time t
            theta_t = self.forward_curve(t)

            # Exact Conditional Mean formula (pulling toward the long-term average price of electricity at time t, theta_t)
            drift = S_prev * np.exp(-self.kappa * self.dt) + theta_t * (1 - np.exp(-self.kappa * self.dt))
            
            # Diffusion (Wiener Process)
            diffusion = exact_std * Z[:, t]

            # Poisson Jumps
            jumps = jump_flags[:, t] * jump_sizes[:, t]
            
            self.simulated_prices[:, t] = drift + diffusion + jumps
            # Example of upper and lower exchange-mandated price bounds
            self.simulated_prices[:, t] = np.clip(self.simulated_prices[:, t], -500.0, 3000.0)
        
        return self.simulated_prices
    
    def calculate_exposure(self):
        """Calculates Cash Flows applying Cannibalization Effect and Yield Curve."""
        sim_prices_z = (self.simulated_prices - self.price_mean) / self.price_std

        # Commercial Logic based on Asset Profile
        # Note: generator receives Fixed, pays Floating
        if "Wind" in self.asset_class:
            # Polynomial Cannibalization (accounting for grid volume saturation)
            capacity_factor = self.alpha_vol + (self.beta_vol * sim_prices_z) + (self.gamma_vol * (sim_prices_z**2))
            # The asset cannot generate less than 0 or more than max capacity
            dynamic_volume = self.max_capacity * capacity_factor
            dynamic_volume = np.clip(dynamic_volume, 0.0, self.max_capacity)
            daily_cash_flow = (self.fixed_price - self.simulated_prices) * (dynamic_volume * 24)

        elif "Solar" in self.asset_class:
            # Capture discount (based on a "deterministic duck curve") is considered
            solar_capture_discount = 0.70 
            captured_price = self.simulated_prices * solar_capture_discount
            # Generation profile (e.g., 8 hours of effective "full-load" generation)
            daily_cash_flow = (self.fixed_price - captured_price) * (self.max_capacity * 8)
        
        elif "BESS" in self.asset_class:
            # A battery relies on the intraday spread
            # We can assume the battery charges during the cheapest 4 hours of any given day.
            # It then discharges during the peak 4 hours.
            charge_price = self.simulated_prices * 0.60
            discharge_price = self.simulated_prices * 1.40
            rte = 0.85 # Round Trip Efficiency (thermal loss)
            
            revenue = (self.max_capacity * 4) * discharge_price * rte
            cost = (self.max_capacity * 4) * charge_price

            daily_cash_flow = revenue - cost

        # Discounted Cash Flow 
        discounted_cf = daily_cash_flow * self.discount_factor.reshape(1, -1)

        # Cumulative Present Value along each path
        cumulative_pv = np.cumsum(discounted_cf, axis=1)
        
        self.expected_mtm = np.mean(cumulative_pv, axis=0)
        self.pfe_95 = np.percentile(cumulative_pv, 5, axis=0)
        
        return self.expected_mtm, self.pfe_95
    
# Run & Plot
model = PPA_Exposure_Model(
    S0=S0, kappa=kappa_base, sigma=sigma_input, lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j,
    fwd_prices=fwd_prices, yield_rates=yield_rates, asset=asset_class,
    alpha_vol=alpha_emp, beta_vol=beta_emp, gamma_vol=gamma_emp, price_mean=p_mean, price_std=p_std,
    max_capacity=48.0, fixed_price=fixed_leg_input, paths=5000
)

model.simulate_prices()
expected_mtm, pfe_95 = model.calculate_exposure()

# Extract final values for the UI metrics
final_mtm = expected_mtm[-1]
final_pfe = pfe_95[-1]

# =====================================================================
# DASHBOARD VISUALIZATION
# =====================================================================
col1, col2, col3 = st.columns(3)
if "BESS" in asset_class:
    col1.metric(label="Asset Trading Strategy", value="Merchant (Spot Arbitrage)")
else:
    col1.metric(label="PPA Strike Price", value=f"€{fixed_leg_input:.2f} / MWh")

col2.metric(label="Final Expected MtM (Year 5)", value=f"€{final_mtm:,.0f}")
col3.metric(label="Final 95% PFE Risk (Year 5)", value=f"€{final_pfe:,.0f}")

st.markdown("### 📉 Cumulative Financial Exposure Profile")
fig = go.Figure()
fig.add_trace(go.Scatter(x=timeline_dates, y=expected_mtm, mode='lines', name='Expected Mark-to-Market', line=dict(color='#00CC96', width=3)))
fig.add_trace(go.Scatter(x=timeline_dates, y=pfe_95, mode='lines', name='95% PFE (Downside Risk)', line=dict(color='#EF553B', width=3, dash='dash')))
# Shade the tail risk zone
fig.add_trace(go.Scatter(x=timeline_dates.tolist() + timeline_dates.tolist()[::-1], y=expected_mtm.tolist() + pfe_95.tolist()[::-1], fill='toself', fillcolor='rgba(239, 85, 59, 0.1)', line=dict(color='rgba(255,255,255,0)'), showlegend=False))

fig.update_layout(xaxis_title="Forward Curve Timeline", yaxis_title="Present Value (€)", hovermode="x unified", template="plotly_dark", legend=dict(orientation="h", y=1.02))
st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# EXPERIMENT LEDGER
# =====================================================================
st.markdown("---")
st.markdown("### 💾 Structuring Scenario Ledger")
if st.button("Log Current Scenario"):
    st.session_state.experiment_history.append({
        "Run": len(st.session_state.experiment_history) + 1,
        "Fixed Leg (€/MWh)": fixed_leg_input,
        "Threshold (Std)": threshold_input,
        "Diffusion Vol (σ)": round(sigma_input, 2),
        "Calibrated Jumps/Yr": round(lambda_j, 1),
        "Final Expected MtM": f"€{final_mtm:,.0f}",
        "Final 95% PFE": f"€{final_pfe:,.0f}"
    })
    st.success("Scenario logged to matrix.")

if st.session_state.experiment_history:
    st.dataframe(pd.DataFrame(st.session_state.experiment_history).set_index("Run"), use_container_width=True)
else:
    st.info("Adjust the sliders on the left and log scenarios to compare risk profiles.")