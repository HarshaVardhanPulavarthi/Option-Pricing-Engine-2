"""
Enhanced Option Pricing Engines
===============================

This package contains implementations of various option pricing models
with support for both European and American options, including Greeks calculations.

Available engines:
- black_scholes: Black-Scholes-Merton model with analytical Greeks
- binomial: Binomial tree model with finite difference Greeks  
- crank_nicolson: Crank-Nicolson finite difference method
"""

__version__ = "2.0.0"
__author__ = "Option Pricing Workbench"

from .black_scholes import bsm_price_greeks
from .binomial import binom_price_greeks  
from .crank_nicolson import cn_price_greeks

__all__ = ["bsm_price_greeks", "binom_price_greeks", "cn_price_greeks"]
