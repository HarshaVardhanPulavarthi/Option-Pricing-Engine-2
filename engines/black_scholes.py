import numpy as np
from scipy.stats import norm
from math import log, sqrt, exp

def bsm_price_greeks(S, K, T, r, sigma, q=0.0):
    """
    Calculate Black-Scholes option prices and Greeks for both calls and puts.

    Parameters:
    S: Current stock price
    K: Strike price  
    T: Time to maturity (years)
    r: Risk-free rate
    sigma: Volatility
    q: Dividend yield (default 0)

    Returns:
    Dictionary with option prices and Greeks
    """
    if T <= 0:
        # Handle expiration case
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

    # Calculate d1 and d2
    d1 = (log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    # Discount factors
    disc_q = exp(-q*T)
    disc_r = exp(-r*T)

    # Option prices
    call_price = disc_q*S*norm.cdf(d1) - disc_r*K*norm.cdf(d2)
    put_price = call_price + disc_r*K - disc_q*S  # Put-call parity

    # Greeks calculations
    # Delta
    delta_call = disc_q * norm.cdf(d1)
    delta_put = delta_call - disc_q

    # Gamma (same for calls and puts)
    gamma = disc_q * norm.pdf(d1) / (S * sigma * sqrt(T))

    # Theta
    theta_call = (-disc_q*S*norm.pdf(d1)*sigma/(2*sqrt(T)) 
                  - r*disc_r*K*norm.cdf(d2) 
                  + q*disc_q*S*norm.cdf(d1)) / 365
    theta_put = (-disc_q*S*norm.pdf(d1)*sigma/(2*sqrt(T)) 
                 + r*disc_r*K*norm.cdf(-d2) 
                 - q*disc_q*S*norm.cdf(-d1)) / 365

    # Vega (same for calls and puts)
    vega = disc_q * S * norm.pdf(d1) * sqrt(T) / 100

    # Rho
    rho_call = K * T * disc_r * norm.cdf(d2) / 100
    rho_put = -K * T * disc_r * norm.cdf(-d2) / 100

    return {
        "Call Price": call_price,
        "Put Price": put_price,
        "Delta": delta_call,  # Using call delta as primary
        "Gamma": gamma,
        "Theta": theta_call,  # Using call theta as primary
        "Vega": vega,
        "Rho": rho_call      # Using call rho as primary
    }
