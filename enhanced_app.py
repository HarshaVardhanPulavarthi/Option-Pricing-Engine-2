import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engines.black_scholes import bsm_price_greeks
from engines.binomial import binom_price_greeks
from engines.crank_nicolson import cn_price_greeks

st.set_page_config(
    page_title="Advanced Option Pricing Workbench", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 Advanced Option Pricing Workbench")
st.markdown("***Compare American vs European Options using Black-Scholes, Binomial Tree, and Crank-Nicolson methods***")

# Sidebar for inputs
st.sidebar.header("📊 Option Parameters")

# NEW FEATURE: American vs European Option Selection
option_style = st.sidebar.selectbox(
    "Option Style",
    ["European", "American"],
    index=0,
    help="European options can only be exercised at expiration. American options can be exercised any time before expiration."
)

# Basic option parameters
S = st.sidebar.number_input("Spot Price (S)", min_value=0.01, value=299.55, step=0.01)
K = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=300.0, step=0.01)
sigma = st.sidebar.slider("Volatility (σ) %", min_value=1.0, max_value=100.0, value=30.0, step=0.5) / 100
r = st.sidebar.slider("Risk-free Rate (r) %", min_value=0.0, max_value=20.0, value=4.0, step=0.1) / 100
T = st.sidebar.number_input("Time to Maturity (T) years", min_value=0.01, max_value=10.0, value=1.0, step=0.01)

# NEW FEATURE: Conditional parameters based on option style
if option_style == "American":
    st.sidebar.subheader("🔧 American Option Parameters")
    q = st.sidebar.slider("Dividend Yield (q) %", min_value=0.0, max_value=15.0, value=0.0, step=0.1) / 100
    st.sidebar.markdown("💡 **Early Exercise Premium:** American options typically cost more due to the flexibility of early exercise")
else:
    q = 0.0
    st.sidebar.info("ℹ️ European options can only be exercised at expiration")

# Model selection
st.sidebar.header("🛠️ Pricing Models")
models = st.sidebar.multiselect(
    "Select Pricing Methods",
    ["Black-Scholes", "Binomial Tree", "Crank-Nicolson"],
    default=["Black-Scholes", "Binomial Tree", "Crank-Nicolson"]
)

# Model-specific parameters
if "Binomial Tree" in models:
    st.sidebar.subheader("🌳 Binomial Tree Settings")
    N = st.sidebar.slider("Number of Steps", min_value=10, max_value=1000, value=500, step=10)

if "Crank-Nicolson" in models:
    st.sidebar.subheader("📐 Crank-Nicolson Settings")
    M = st.sidebar.slider("Stock Price Grid Points", min_value=50, max_value=300, value=150, step=10)
    N_fd = st.sidebar.slider("Time Steps", min_value=50, max_value=300, value=150, step=10)

# Calculation button
calculate = st.sidebar.button("🚀 Calculate Options", type="primary")

if calculate and models:
    with st.spinner("Calculating option prices and Greeks..."):
        results = {}

        try:
            # Calculate using selected models
            if "Black-Scholes" in models:
                # Note: BS inherently European, but we'll adapt for comparison
                bs_results = bsm_price_greeks(S, K, T, r, sigma, q)
                if option_style == "American":
                    # Add early exercise premium estimation (simplified)
                    early_exercise_premium = max(0, 0.02 * bs_results["Call Price"])
                    bs_results["Call Price"] += early_exercise_premium
                    bs_results["Put Price"] += early_exercise_premium
                results["Black-Scholes"] = bs_results

            if "Binomial Tree" in models:
                results["Binomial Tree"] = binom_price_greeks(
                    S, K, T, r, sigma, q, N, option_style.lower()
                )

            if "Crank-Nicolson" in models:
                results["Crank-Nicolson"] = cn_price_greeks(
                    S, K, T, r, sigma, q, M, N_fd, option_style.lower()
                )

            # Create results DataFrame
            df = pd.DataFrame(results).T

            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Comparison", "🎯 Greeks Analysis", "📊 Detailed Results", "📉 Sensitivity Analysis"])

            with tab1:
                st.subheader(f"💰 {option_style} Option Prices")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📞 Call Options")
                    call_data = df[["Call Price"]].reset_index()
                    call_fig = px.bar(call_data, x="index", y="Call Price", 
                                    title=f"{option_style} Call Option Prices",
                                    color="Call Price", color_continuous_scale="viridis")
                    st.plotly_chart(call_fig, use_container_width=True)

                with col2:
                    st.markdown("#### 📞 Put Options")
                    put_data = df[["Put Price"]].reset_index()
                    put_fig = px.bar(put_data, x="index", y="Put Price",
                                   title=f"{option_style} Put Option Prices", 
                                   color="Put Price", color_continuous_scale="plasma")
                    st.plotly_chart(put_fig, use_container_width=True)

            with tab2:
                st.subheader("🎯 Greeks Comparison")

                # Greeks radar chart
                greeks_cols = ["Delta", "Gamma", "Theta", "Vega", "Rho"]
                available_greeks = [col for col in greeks_cols if col in df.columns]

                if available_greeks:
                    fig = go.Figure()

                    for model in df.index:
                        values = [df.loc[model, greek] for greek in available_greeks]
                        values += [values[0]]  # Close the radar chart

                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=available_greeks + [available_greeks[0]],
                            fill='toself',
                            name=model,
                            line_color='rgb' + str(tuple(np.random.randint(0, 255, 3)))
                        ))

                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True)),
                        showlegend=True,
                        title=f"Greeks Comparison - {option_style} Options"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Individual Greeks charts
                for greek in available_greeks:
                    if greek in df.columns:
                        greek_data = df[[greek]].reset_index()
                        fig = px.line(greek_data, x="index", y=greek,
                                    title=f"{greek} Comparison", markers=True)
                        st.plotly_chart(fig, use_container_width=True)

            with tab3:
                st.subheader("📊 Detailed Pricing Results")
                st.dataframe(df.style.format("{:.6f}"), use_container_width=True)

                # Summary statistics
                st.subheader("📈 Summary Statistics")
                summary = df.describe()
                st.dataframe(summary.style.format("{:.6f}"), use_container_width=True)

                # Download results
                csv = df.to_csv()
                st.download_button(
                    label="💾 Download Results as CSV",
                    data=csv,
                    file_name=f"{option_style.lower()}_option_results.csv",
                    mime="text/csv"
                )

            with tab4:
                st.subheader("📉 Sensitivity Analysis")
                st.info("🔬 Analyze how option prices change with different parameters")

                # Sensitivity to spot price
                spot_range = np.linspace(S*0.8, S*1.2, 20)
                sensitivity_results = []

                for spot in spot_range:
                    if "Black-Scholes" in models:
                        bs_temp = bsm_price_greeks(spot, K, T, r, sigma, q)
                        sensitivity_results.append({
                            "Spot": spot,
                            "Model": "Black-Scholes", 
                            "Call": bs_temp["Call Price"],
                            "Put": bs_temp["Put Price"]
                        })

                if sensitivity_results:
                    sens_df = pd.DataFrame(sensitivity_results)

                    fig = make_subplots(rows=1, cols=2, subplot_titles=("Call Price Sensitivity", "Put Price Sensitivity"))

                    for model in sens_df["Model"].unique():
                        model_data = sens_df[sens_df["Model"] == model]
                        fig.add_trace(go.Scatter(x=model_data["Spot"], y=model_data["Call"], 
                                               name=f"{model} Call", mode="lines"), row=1, col=1)
                        fig.add_trace(go.Scatter(x=model_data["Spot"], y=model_data["Put"], 
                                               name=f"{model} Put", mode="lines"), row=1, col=2)

                    fig.update_layout(title="Price Sensitivity to Spot Price")
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error in calculation: {str(e)}")
            st.error("Please check your parameters and try again.")

else:
    # Display information when no calculation is performed
    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        ### 📚 About Option Styles

        **European Options:**
        - Can only be exercised at expiration
        - Generally cheaper due to limited flexibility
        - Most index options are European-style

        **American Options:**
        - Can be exercised any time before expiration
        - More expensive due to early exercise flexibility
        - Most stock options are American-style
        """)

    with col2:
        st.info("""
        ### 🛠️ Pricing Methods

        **Black-Scholes Model:**
        - Closed-form analytical solution
        - Fast computation
        - Assumes European-style exercise

        **Binomial Tree:**
        - Discrete time model
        - Handles American options naturally
        - Good for understanding option behavior

        **Crank-Nicolson:**
        - Finite difference method
        - High accuracy for both styles
        - Stable numerical solution
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
Built with ❤️ using Streamlit | Advanced Option Pricing Workbench v2.0
</div>
""", unsafe_allow_html=True)
