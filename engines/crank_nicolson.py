import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from math import exp, sqrt, log

def cn_price_greeks(S0, K, T, r, sigma, q=0.0, M=150, N=150, option_style="european"):
    """
    Calculate option prices and Greeks using Crank-Nicolson finite difference method.
    Supports both European and American options.

    Parameters:
    S0: Current stock price
    K: Strike price
    T: Time to maturity (years)
    r: Risk-free rate
    sigma: Volatility
    q: Dividend yield
    M: Number of stock price grid points
    N: Number of time steps
    option_style: "european" or "american"

    Returns:
    Dictionary with option prices and Greeks
    """
    if T <= 0:
        call_price = max(S0 - K, 0)
        put_price = max(K - S0, 0)
        return {
            "Call Price": call_price,
            "Put Price": put_price,
            "Delta": 1.0 if call_price > 0 else 0.0,
            "Gamma": 0.0,
            "Theta": 0.0,
            "Vega": 0.0,
            "Rho": 0.0
        }

    # Grid parameters
    S_max = 3 * max(S0, K)  # Maximum stock price
    dS = S_max / M
    dt = T / N

    # Stock price grid
    S = np.linspace(0, S_max, M+1)

    def solve_pde(is_call=True):
        """Solve PDE for call or put option"""

        # Initial condition (payoff at maturity)
        if is_call:
            V = np.maximum(S - K, 0)
        else:
            V = np.maximum(K - S, 0)

        # Coefficients for finite difference scheme
        alpha = np.zeros(M+1)
        beta = np.zeros(M+1)
        gamma = np.zeros(M+1)

        for i in range(1, M):
            alpha[i] = 0.25 * dt * (sigma**2 * i**2 - (r - q) * i)
            beta[i] = -0.5 * dt * (sigma**2 * i**2 + r)
            gamma[i] = 0.25 * dt * (sigma**2 * i**2 + (r - q) * i)

        # Build tridiagonal matrices for Crank-Nicolson
        # A * V^{n+1} = B * V^n

        # Matrix A (implicit part)
        A = sparse.diags(
            [alpha[2:M], 1 + beta[1:M], gamma[1:M-1]], 
            [-1, 0, 1], 
            shape=(M-1, M-1)
        ).tocsr()

        # Matrix B (explicit part)
        B = sparse.diags(
            [-alpha[2:M], 1 - beta[1:M], -gamma[1:M-1]], 
            [-1, 0, 1], 
            shape=(M-1, M-1)
        ).tocsr()

        # Time stepping
        for n in range(N):
            # Right-hand side
            rhs = B.dot(V[1:M])

            # Boundary conditions
            if is_call:
                # V(0,t) = 0, V(S_max,t) = S_max - K*exp(-r*(T-t))
                rhs[0] -= alpha[1] * 0  # Left boundary
                rhs[-1] -= gamma[M-1] * (S_max - K * exp(-r * (T - n*dt)))  # Right boundary
            else:
                # V(0,t) = K*exp(-r*(T-t)), V(S_max,t) = 0
                rhs[0] -= alpha[1] * (K * exp(-r * (T - n*dt)))  # Left boundary  
                rhs[-1] -= gamma[M-1] * 0  # Right boundary

            # Solve linear system
            V_new = spsolve(A, rhs)

            # Apply American option constraint (early exercise)
            if option_style.lower() == "american":
                if is_call:
                    V[1:M] = np.maximum(V_new, S[1:M] - K)
                else:
                    V[1:M] = np.maximum(V_new, K - S[1:M])
            else:
                V[1:M] = V_new

        return V

    # Solve for call and put options
    V_call = solve_pde(is_call=True)
    V_put = solve_pde(is_call=False)

    # Find the option prices at current stock price S0
    idx = np.argmin(np.abs(S - S0))
    call_price = V_call[idx]
    put_price = V_put[idx]

    # Calculate Greeks using finite difference on the grid
    h = dS

    # Delta (first derivative w.r.t. S)
    if idx > 0 and idx < len(S) - 1:
        delta = (V_call[idx+1] - V_call[idx-1]) / (2 * h)
    else:
        delta = 0.0

    # Gamma (second derivative w.r.t. S)  
    if idx > 0 and idx < len(S) - 1:
        gamma = (V_call[idx+1] - 2*V_call[idx] + V_call[idx-1]) / (h**2)
    else:
        gamma = 0.0

    # Calculate Theta, Vega, Rho using parameter bumping
    h_t = 1/365  # 1 day
    h_vol = 0.01  # 1%
    h_r = 0.0001  # 0.01%

    # Theta
    try:
        if T > h_t:
            theta_result = cn_price_greeks(S0, K, T - h_t, r, sigma, q, M, N, option_style)
            theta = (theta_result["Call Price"] - call_price) / h_t
        else:
            theta = 0.0
    except:
        theta = 0.0

    # Vega
    try:
        vega_result = cn_price_greeks(S0, K, T, r, sigma + h_vol, q, M, N, option_style)
        vega = (vega_result["Call Price"] - call_price) / (h_vol * 100)
    except:
        vega = 0.0

    # Rho
    try:
        rho_result = cn_price_greeks(S0, K, T, r + h_r, sigma, q, M, N, option_style)
        rho = (rho_result["Call Price"] - call_price) / (h_r * 100)
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


def cn_detailed_solution(S0, K, T, r, sigma, q=0.0, M=150, N=150, option_style="european"):
    """
    Return detailed solution including full price grid for visualization.
    """
    # This is a simplified version - full implementation would return
    # the complete solution grid for advanced analysis
    result = cn_price_greeks(S0, K, T, r, sigma, q, M, N, option_style)

    # Add some additional analysis
    S_max = 3 * max(S0, K)
    S_grid = np.linspace(0, S_max, M+1)

    result["S_grid"] = S_grid
    result["Method"] = "Crank-Nicolson"
    result["Grid_Points"] = M
    result["Time_Steps"] = N
    result["Option_Style"] = option_style

    return result
