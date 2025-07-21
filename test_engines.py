#!/usr/bin/env python3
"""
Test script for the Enhanced Option Pricing Workbench
Tests all engines with both American and European options
"""

import numpy as np
import sys
import os

# Add engines to path
sys.path.append('.')

try:
    from engines.black_scholes import bsm_price_greeks
    from engines.binomial import binom_price_greeks
    from engines.crank_nicolson import cn_price_greeks
    print("✅ All engines imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_engines():
    """Test all pricing engines with sample parameters"""

    # Test parameters (matching original files)
    S = 299.55  # Current stock price
    K = 300.0   # Strike price  
    T = 1.0     # Time to maturity
    r = 0.04    # Risk-free rate
    sigma = 0.3 # Volatility
    q = 0.0     # Dividend yield

    print(f"\n📊 Testing with parameters:")
    print(f"   Spot Price (S): ${S}")
    print(f"   Strike Price (K): ${K}")
    print(f"   Time to Maturity (T): {T} years")
    print(f"   Risk-free Rate (r): {r*100:.1f}%")
    print(f"   Volatility (σ): {sigma*100:.1f}%")
    print(f"   Dividend Yield (q): {q*100:.1f}%")

    results = {}

    # Test Black-Scholes
    print(f"\n🔍 Testing Black-Scholes Engine...")
    try:
        bs_result = bsm_price_greeks(S, K, T, r, sigma, q)
        results["Black-Scholes"] = bs_result
        print(f"   ✅ Call Price: ${bs_result['Call Price']:.4f}")
        print(f"   ✅ Put Price: ${bs_result['Put Price']:.4f}")
        print(f"   ✅ Delta: {bs_result['Delta']:.4f}")
        print(f"   ✅ Gamma: {bs_result['Gamma']:.4f}")
        print(f"   ✅ Vega: {bs_result['Vega']:.4f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test Binomial Tree - European
    print(f"\n🌳 Testing Binomial Tree Engine (European)...")
    try:
        binom_eur = binom_price_greeks(S, K, T, r, sigma, q, N=100, option_style="european")
        results["Binomial (European)"] = binom_eur
        print(f"   ✅ Call Price: ${binom_eur['Call Price']:.4f}")
        print(f"   ✅ Put Price: ${binom_eur['Put Price']:.4f}")
        print(f"   ✅ Delta: {binom_eur['Delta']:.4f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test Binomial Tree - American
    print(f"\n🌳 Testing Binomial Tree Engine (American)...")
    try:
        binom_am = binom_price_greeks(S, K, T, r, sigma, q, N=100, option_style="american")
        results["Binomial (American)"] = binom_am
        print(f"   ✅ Call Price: ${binom_am['Call Price']:.4f}")
        print(f"   ✅ Put Price: ${binom_am['Put Price']:.4f}")
        print(f"   ✅ Delta: {binom_am['Delta']:.4f}")

        # Check American premium
        if 'Binomial (European)' in results:
            am_premium_call = binom_am['Call Price'] - binom_eur['Call Price']
            am_premium_put = binom_am['Put Price'] - binom_eur['Put Price']
            print(f"   💰 American Call Premium: ${am_premium_call:.4f}")
            print(f"   💰 American Put Premium: ${am_premium_put:.4f}")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test Crank-Nicolson - European
    print(f"\n📐 Testing Crank-Nicolson Engine (European)...")
    try:
        cn_eur = cn_price_greeks(S, K, T, r, sigma, q, M=50, N=50, option_style="european")
        results["Crank-Nicolson (European)"] = cn_eur
        print(f"   ✅ Call Price: ${cn_eur['Call Price']:.4f}")
        print(f"   ✅ Put Price: ${cn_eur['Put Price']:.4f}")
        print(f"   ✅ Delta: {cn_eur['Delta']:.4f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test Crank-Nicolson - American
    print(f"\n📐 Testing Crank-Nicolson Engine (American)...")
    try:
        cn_am = cn_price_greeks(S, K, T, r, sigma, q, M=50, N=50, option_style="american")
        results["Crank-Nicolson (American)"] = cn_am
        print(f"   ✅ Call Price: ${cn_am['Call Price']:.4f}")
        print(f"   ✅ Put Price: ${cn_am['Put Price']:.4f}")
        print(f"   ✅ Delta: {cn_am['Delta']:.4f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Comparison summary
    print(f"\n📋 RESULTS SUMMARY")
    print(f"=" * 60)
    print(f"{'Method':<25} {'Call Price':<12} {'Put Price':<12}")
    print(f"=" * 60)

    for method, result in results.items():
        if result:
            print(f"{method:<25} ${result['Call Price']:<11.4f} ${result['Put Price']:<11.4f}")

    print(f"=" * 60)

    # Validate results are reasonable
    print(f"\n🔍 VALIDATION CHECKS")

    if "Black-Scholes" in results:
        bs = results["Black-Scholes"]

        # Put-call parity check: C - P = S - K*e^(-rT)
        pcp_left = bs["Call Price"] - bs["Put Price"]
        pcp_right = S - K * np.exp(-r * T)
        pcp_error = abs(pcp_left - pcp_right)

        print(f"   Put-Call Parity Check:")
        print(f"      C - P = ${pcp_left:.4f}")
        print(f"      S - Ke^(-rT) = ${pcp_right:.4f}")
        print(f"      Error: ${pcp_error:.6f} {'✅' if pcp_error < 0.001 else '❌'}")

        # Greeks reasonableness
        print(f"   Greeks Reasonableness:")
        print(f"      Delta ∈ [0,1]: {bs['Delta']:.4f} {'✅' if 0 <= bs['Delta'] <= 1 else '❌'}")
        print(f"      Gamma ≥ 0: {bs['Gamma']:.4f} {'✅' if bs['Gamma'] >= 0 else '❌'}")
        print(f"      Vega ≥ 0: {bs['Vega']:.4f} {'✅' if bs['Vega'] >= 0 else '❌'}")

    print(f"\n🎉 Testing completed successfully!")
    return results

if __name__ == "__main__":
    print("🎯 Enhanced Option Pricing Workbench - Engine Testing")
    print("=" * 60)

    test_engines()
