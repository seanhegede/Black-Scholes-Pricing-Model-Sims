import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time

def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate European call option price using Black-Scholes-Merton formula.
    
    Parameters:
    S: Stock price (can be array)
    K: Strike price
    T: Time to maturity in years
    r: Risk-free rate (annual)
    sigma: Volatility (annual)
    
    Returns:
    Call option price
    """
    if T == 0:
        return np.maximum(S - K, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def create_plot(S_vals, K, T, r, sigma):
    """
    Create the call option price plot for a given time to maturity.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot intrinsic value (payoff at maturity)
    intrinsic = np.maximum(S_vals - K, 0)
    ax.plot(S_vals, intrinsic, 'k--', linewidth=2, label='Intrinsic Value (T=0)', alpha=0.5)
    
    # Calculate and plot call prices for current T
    call_prices = black_scholes_call(S_vals, K, T, r, sigma)
    ax.plot(S_vals, call_prices, 'b-', linewidth=2.5, label='Call Price')
    
    # Add strike price line
    ax.axvline(x=K, color='r', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Strike K=${K}')
    
    # Add time text
    ax.text(0.58, 0.02, f'Time to Maturity: T = {T:.3f} years', 
            transform=ax.transAxes, fontsize=12, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Set labels and title
    ax.set_xlabel('Stock Price (S)', fontsize=12)
    ax.set_ylabel('Call Option Price (C)', fontsize=12)
    ax.set_title(f'Call Option Price vs Stock Price\nK=${K}, r={r*100}%, σ={sigma*100}%', 
                 fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    # Set axis limits
    ax.set_xlim(50, 150)
    ax.set_ylim(0, 60)
    
    plt.tight_layout()
    return fig

# Streamlit App
st.set_page_config(page_title="Black-Scholes Call Option Animator", layout="wide")

st.title("📈 Black-Scholes Call Option Price Animation")
st.markdown("Watch how the call option price evolves as time to maturity changes")

# Sidebar for parameters
st.sidebar.header("Option Parameters")

K = st.sidebar.number_input("Strike Price (K)", min_value=50.0, max_value=150.0, value=100.0, step=5.0)
r = st.sidebar.slider("Risk-Free Rate (r)", min_value=0.0, max_value=0.20, value=0.05, step=0.01, format="%.2f")
sigma = st.sidebar.slider("Volatility (σ)", min_value=0.05, max_value=0.50, value=0.20, step=0.05, format="%.2f")

st.sidebar.header("Animation Settings")
T_max = st.sidebar.slider("Maximum Time to Maturity (years)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
num_frames = st.sidebar.slider("Number of Frames", min_value=20, max_value=200, value=100, step=10)
animation_speed = st.sidebar.slider("Animation Speed (ms per frame)", min_value=20, max_value=200, value=50, step=10)

# Stock price range
S_vals = np.linspace(50, 150, 200)

# Time to maturity values for animation
T_vals = np.linspace(0, T_max, num_frames)

# Animation controls
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    animate = st.button("▶️ Start Animation", use_container_width=True)
with col2:
    stop = st.button("⏸️ Stop", use_container_width=True)
with col3:
    manual_mode = st.checkbox("Manual Mode (use slider)")

# Initialize session state
if 'is_animating' not in st.session_state:
    st.session_state.is_animating = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0

# Handle button clicks
if animate:
    st.session_state.is_animating = True
    st.session_state.current_frame = 0

if stop:
    st.session_state.is_animating = False

# Manual slider for frame selection
if manual_mode:
    st.session_state.is_animating = False
    st.session_state.current_frame = st.slider(
        "Select Frame", 
        min_value=0, 
        max_value=num_frames-1, 
        value=st.session_state.current_frame
    )

# Plot placeholder
plot_placeholder = st.empty()

# Display current parameters
with st.expander("ℹ️ Current Parameters"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Strike Price", f"${K:.2f}")
    with col2:
        st.metric("Risk-Free Rate", f"{r*100:.2f}%")
    with col3:
        st.metric("Volatility", f"{sigma*100:.2f}%")
    with col4:
        st.metric("Max Time", f"{T_max:.2f} years")

# Animation loop
if st.session_state.is_animating:
    for frame in range(st.session_state.current_frame, num_frames):
        if not st.session_state.is_animating:
            break
            
        T = T_vals[frame]
        fig = create_plot(S_vals, K, T, r, sigma)
        plot_placeholder.pyplot(fig)
        plt.close(fig)
        
        st.session_state.current_frame = frame
        time.sleep(animation_speed / 1000.0)
        
        # Loop back to start
        if frame == num_frames - 1:
            st.session_state.current_frame = 0
else:
    # Display current frame
    T = T_vals[st.session_state.current_frame]
    fig = create_plot(S_vals, K, T, r, sigma)
    plot_placeholder.pyplot(fig)
    plt.close(fig)

# Information section
st.markdown("---")
st.markdown("""
### About the Black-Scholes Model

The **Black-Scholes-Merton model** prices European call options based on:
- **S**: Current stock price
- **K**: Strike price (exercise price)
- **T**: Time to maturity
- **r**: Risk-free interest rate
- **σ**: Volatility of the underlying stock

**Key Observations:**
- As **T → 0**, the call price converges to the intrinsic value: max(S - K, 0)
- Higher **T** increases option value (more time for favorable price movements)
- The option has **time value** when T > 0, even if currently out-of-the-money
""")