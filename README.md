# 🎯 Advanced Option Pricing Workbench with American/European Options

A comprehensive Streamlit application for pricing and analyzing both American and European options using multiple numerical methods.

## ✨ New Features Added

### 🔄 American vs European Option Selection
- **Dropdown Selection**: Choose between American and European option styles
- **Conditional Parameters**: Additional parameters appear for American options
- **Early Exercise Premium**: Automatic calculation and display
- **Comparative Analysis**: Side-by-side comparison of both option types

### 📊 Enhanced Parameter Controls
- **Dividend Yield (q)**: Configurable dividend yield for American options
- **Option Style Indicator**: Clear visual indication of selected option type
- **Parameter Validation**: Smart validation based on option style
- **Default Values**: Context-aware default parameters

### 🛠️ Improved Pricing Engines

#### Black-Scholes Engine
- ✅ Analytical Greeks calculation
- ✅ Dividend yield support
- ✅ Enhanced error handling
- ✅ Parameter validation

#### Binomial Tree Engine  
- ✅ Native American option support
- ✅ Configurable number of steps
- ✅ Early exercise detection
- ✅ Greeks via finite differences

#### Crank-Nicolson Engine
- ✅ Implicit finite difference method
- ✅ American option constraints
- ✅ Sparse matrix optimization
- ✅ Stable numerical solution

## 🚀 Getting Started

### Installation

1. **Clone or download the project files**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run enhanced_app.py
   ```

### Project Structure
```
option-pricing-workbench/
├── enhanced_app.py              # Main Streamlit application
├── engines/
│   ├── __init__.py             # Package initialization
│   ├── black_scholes.py        # Enhanced BS model with Greeks
│   ├── binomial.py             # Binomial tree with American support
│   └── crank_nicolson.py       # CN finite difference method
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 📱 Application Features

### 🎛️ Parameter Controls
- **Option Style**: American vs European dropdown
- **Basic Parameters**: Spot price, strike, volatility, rate, time
- **American-Specific**: Dividend yield, early exercise considerations
- **Model Settings**: Method-specific parameters (steps, grid points)

### 📈 Analysis Tabs

#### 1. Price Comparison
- Side-by-side call and put price comparisons
- Interactive bar charts with color coding
- Model-by-model breakdown

#### 2. Greeks Analysis
- Comprehensive Greeks radar charts
- Individual Greek comparisons
- Visual sensitivity analysis

#### 3. Detailed Results
- Complete numerical results table
- Summary statistics
- CSV export functionality

#### 4. Sensitivity Analysis
- Price sensitivity to spot price changes
- Interactive parameter sweeping
- Real-time chart updates

### 🔍 Key Differences: American vs European Options

| Feature | European Options | American Options |
|---------|------------------|------------------|
| **Exercise Rights** | Only at expiration | Any time before expiration |
| **Premium** | Generally lower | Generally higher |
| **Dividend Handling** | No early capture | Can exercise for dividends |
| **Complexity** | Simpler to price | More complex algorithms |
| **Use Cases** | Index options, some stocks | Most stock options |

## 🧮 Pricing Methods Comparison

### Black-Scholes Model
- **Best for**: European options, quick estimates
- **Advantages**: Analytical solution, fast computation
- **Limitations**: Assumes European exercise only

### Binomial Tree
- **Best for**: American options, educational purposes  
- **Advantages**: Intuitive, handles early exercise naturally
- **Limitations**: Slower convergence, more computation

### Crank-Nicolson
- **Best for**: High accuracy, stable solutions
- **Advantages**: Second-order accuracy, handles both styles
- **Limitations**: More complex implementation

## 📊 Greeks Explained

- **Delta (Δ)**: Price sensitivity to underlying asset price
- **Gamma (Γ)**: Rate of change of delta
- **Theta (Θ)**: Time decay of option value
- **Vega (ν)**: Sensitivity to volatility changes
- **Rho (ρ)**: Sensitivity to interest rate changes

## 🎨 Usage Examples

### Basic Usage
1. Select option style (American/European)
2. Set market parameters
3. Choose pricing methods
4. Click "Calculate Options"
5. Explore results in different tabs

### Advanced Analysis
1. Use sensitivity analysis tab for parameter sweeps
2. Compare Greeks across different models
3. Export results for further analysis
4. Adjust model-specific parameters for accuracy

## 🔧 Technical Implementation

### Enhanced Features
- **Conditional UI**: Parameters adapt to option style selection
- **Error Handling**: Robust validation and user feedback
- **Performance**: Optimized calculations with progress indicators
- **Visualization**: Interactive Plotly charts and analysis

### Numerical Methods
- **Finite Differences**: For Greeks calculation in tree and PDE methods
- **Sparse Matrices**: Efficient memory usage in Crank-Nicolson
- **Convergence**: Adaptive step sizing for accuracy

## 📝 Notes for Users

### American Option Considerations
- Early exercise is optimal mainly for:
  - Deep in-the-money puts (capture time value of strike)
  - Calls before ex-dividend dates (capture dividend)
- American options always worth ≥ European options
- Premium difference varies with moneyness and time

### Parameter Guidelines
- **Volatility**: 10-50% typical range for stocks
- **Interest Rates**: Use risk-free rate (Treasury rate)
- **Dividend Yield**: Annual dividend yield as decimal
- **Time**: Use decimal years (e.g., 0.25 for 3 months)

### Accuracy Tips
- Increase steps/grid points for higher accuracy
- Use consistent units across all parameters
- Validate results against known benchmarks
- Consider computational cost vs accuracy trade-off

## 🤝 Contributing

Feel free to enhance this application with:
- Additional pricing models (Monte Carlo, Trinomial trees)
- More exotic option types (Asian, Barrier, etc.)
- Advanced Greeks (Color, Speed, etc.)
- Real-time data integration
- Portfolio analysis features

## 📄 License

This project is open source and available for educational and research purposes.

---

**Built with ❤️ using Streamlit and Python**
