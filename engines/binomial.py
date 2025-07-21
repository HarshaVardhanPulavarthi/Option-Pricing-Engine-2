import numpy as np
from math import exp, sqrt, log

def binom_price_greeks(S, K, T, r, sigma, q=0.0, N=500, option_style="european"):
    """
    Calculate option prices and Greeks using Binomial Tree method.
    Supports both European and American options.

    Parameters:
    S: Current stock price
    K: Strike price
    T: Time to maturity (years)
    r: Risk-free rate
    sigma: Volatility
    q: Dividend yield
    N: Number of time steps
    option_style: "european" or "american"

    Returns:
    Dictionary with option prices and Greeks
    """
    if T <= 0:
        call_price = max(S - K, 0)
        put_price = max(K - S, 0)
        return {
            "Call Price": call_price,
            "Put Price": put_price,
            "Delta": 1.0 if call_price > 0 else 0.0,
            "Gamma": 0.0,
            "Theta": 0.0,
            "Vega": 0.0,
            "Rho": 0.0
        }

    # Time step
    dt = T / N

    # Up and down factors
    u = exp(sigma * sqrt(dt))
    d = 1 / u

    # Risk-neutral probability
    p = (exp((r - q) * dt) - d) / (u - d)

    # Discount factor
    disc = exp(-r * dt)

    def binomial_option_price(is_call=True):
        """Calculate option price using binomial tree"""
        # Initialize asset prices at maturity
        ST = np.array([S * (u ** j) * (d ** (N - j)) for j in range(N + 1)])

        # Payoff at maturity
        if is_call:
            option_values = np.maximum(ST - K, 0)
        else:
            option_values = np.maximum(K - ST, 0)

        # Backward induction
        for i in range(N-1, -1, -1):
            ST = ST[:i+1] / u  # Move to previous step's prices
            option_values = disc * (p * option_values[1:i+2] + (1 - p) * option_values[:i+1])

            # Early exercise check for American options
            if option_style.lower() == "american":
                if is_call:
                    option_values = np.maximum(option_values, ST - K)
                else:
                    option_values = np.maximum(option_values, K - ST)

        return option_values[0]

    # Calculate call and put prices
    call_price = binomial_option_price(is_call=True)
    put_price = binomial_option_price(is_call=False)

    # Calculate Greeks using finite difference approximation
    h_s = 0.01 * S  # 1% of stock price for Delta and Gamma
    h_t = 1/365     # 1 day for Theta
    h_vol = 0.01    # 1% for Vega
    h_r = 0.0001    # 0.01% for Rho

    # Delta (finite difference)
    try:
        call_up = binom_price_greeks(S + h_s, K, T, r, sigma, q, N, option_style)["Call Price"]
        call_down = binom_price_greeks(S - h_s, K, T, r, sigma, q, N, option_style)["Call Price"]
        delta = (call_up - call_down) / (2 * h_s)
    except:
        delta = 0.0

    # Gamma (second derivative)
    try:
        gamma = (call_up - 2*call_price + call_down) / (h_s**2)
    except:
        gamma = 0.0

    # Theta
    try:
        if T > h_t:
            call_t = binom_price_greeks(S, K, T - h_t, r, sigma, q, N, option_style)["Call Price"]
            theta = (call_t - call_price) / h_t
        else:
            theta = 0.0
    except:
        theta = 0.0

    # Vega
    try:
        call_vol = binom_price_greeks(S, K, T, r, sigma + h_vol, q, N, option_style)["Call Price"]
        vega = (call_vol - call_price) / (h_vol * 100)  # Scale to 1% vol change
    except:
        vega = 0.0

    # Rho
    try:
        call_r = binom_price_greeks(S, K, T, r + h_r, sigma, q, N, option_style)["Call Price"]
        rho = (call_r - call_price) / (h_r * 100)  # Scale to 1% rate change
    except:
        rho = 0.0

    return {
        "Call Price": call_price,
        "Put Price": put_price,
        "Delta": delta,
        "Gamma": gamma,
        "Theta": theta,
        "Vega": vega,
        "Rho": rho
    }


def american_binomial_detailed(S, K, T, r, sigma, q=0.0, N=500, is_call=True):
    """
    Detailed American option pricing with exercise boundary information.
    """
    dt = T / N
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    p = (exp((r - q) * dt) - d) / (u - d)
    disc = exp(-r * dt)

    # Store the tree for analysis
    tree = np.zeros((N+1, N+1))
    exercise_boundary = []

    # Initialize at maturity
    for j in range(N+1):
        S_T = S * (u ** j) * (d ** (N - j))
        if is_call:
            tree[N, j] = max(S_T - K, 0)
        else:
            tree[N, j] = max(K - S_T, 0)

    # Backward induction
    for i in range(N-1, -1, -1):
        early_exercise_nodes = 0
        for j in range(i+1):
            S_i = S * (u ** j) * (d ** (i - j))

            # European value (continuation)
            european_value = disc * (p * tree[i+1, j+1] + (1-p) * tree[i+1, j])

            # Intrinsic value (immediate exercise)
            if is_call:
                intrinsic_value = max(S_i - K, 0)
            else:
                intrinsic_value = max(K - S_i, 0)

            # American option value
            tree[i, j] = max(european_value, intrinsic_value)

            # Check if early exercise is optimal
            if intrinsic_value > european_value:
                early_exercise_nodes += 1

        exercise_boundary.append(early_exercise_nodes)

    return tree[0, 0], exercise_boundary
