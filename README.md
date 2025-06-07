# Conditional Value at Risk (CVaR) Optimization for Portfolio Allocation

## Overview

This project demonstrates how quantitative mathematical techniques can be used to optimize financial portfolios by minimizing Conditional Value at Risk (CVaR), a robust measure of downside risk. The implementation avoids reliance on machine learning-based predictions, which often introduce volatility and model uncertainty, and instead focuses on deterministic, optimization-based risk control. 

The goal of this research is to contribute a mathematically rigorous and interpretable framework for portfolio allocation that aligns with the risk-averse practices of quantitative finance firms. The project serves as an educational bridge into core quantitative finance topics such as convex optimization, gradient descent, and statistical tail risk modeling.

This work supports a larger research agenda that explores stable and explainable strategies for managing financial risk, particularly in asset classes like REITs that are sensitive to market shocks but difficult to predict.

---

## Research Focus

**Research Question:**  
How can we minimize Conditional Value at Risk (CVaR) using mathematical optimization to construct a risk-averse portfolio across multiple assets?

This question shifts the focus away from forecasting returns and toward designing a robust portfolio that explicitly minimizes exposure to tail losses. The model is built entirely on quantifiable risk, not uncertain price movement predictions.

---

## Key Concepts

### 1. Conditional Value at Risk (CVaR)

CVaR measures the expected loss in the worst-case scenarios beyond a certain confidence threshold (e.g., 95%). While Value at Risk (VaR) captures the loss threshold itself, CVaR goes a step further by quantifying the **average** of losses in the tail, making it a more comprehensive and conservative risk measure.

Mathematically, for a loss distribution \( L \), the CVaR at confidence level \( \beta \) is defined as:

\[
\text{CVaR}_\beta = \mathbb{E}[L \mid L > \text{VaR}_\beta]
\]

### 2. Optimization Problem

We minimize the following objective function:

\[
F_\beta(x, \alpha) = \alpha + \frac{1}{(1 - \beta)N} \sum_{i=1}^{N} \max( -x^T r_i - \alpha, 0 )
\]

Where:
- \( x \): portfolio weights (decision variable)
- \( \alpha \): auxiliary variable representing VaR
- \( r_i \): return vector for asset scenario \( i \)
- \( N \): total number of return samples
- \( \beta \): confidence level (e.g., 0.95)

### 3. Gradient Descent Optimization

We use gradient descent to minimize \( F_\beta(x, \alpha) \), updating both portfolio weights and the VaR approximation iteratively. The weights are constrained to lie on the simplex (non-negative and sum to 1), enforcing realistic portfolio constraints.

---

## Implementation Details

### Libraries Used
- NumPy for numerical operations
- Matplotlib for visualizing convergence
- Jupyter Notebook for interactive development

### Data

Synthetic return data is generated using a normal distribution with:
- Mean return: 0.001
- Standard deviation: 0.02
- Number of assets: 3
- Sample days: 1000

This setup mimics daily return patterns seen in diversified portfolios.

### Structure

The notebook is organized as follows:
1. Data generation and CVaR confidence level setup
2. Definition of the CVaR objective and gradient functions
3. Initialization of weights and VaR
4. Gradient descent loop over epochs
5. Visualization of convergence (CVaR over time)

---

## Why This Research Matters

- **Robustness:** By removing reliance on future predictions, this method avoids overfitting and generalizes well to new data.
- **Interpretability:** The CVaR framework is mathematically transparent, making it easier to audit and justify in high-stakes financial contexts.
- **Industry Alignment:** Risk minimization via CVaR is a method used by hedge funds, banks, and asset managers under regulatory frameworks such as Basel III and Solvency II.
- **Pedagogical Value:** The project provides a hands-on entry point into quantitative finance for researchers with programming experience but limited exposure to formal financial mathematics.

---

## Next Steps

- Expand to include real market data (e.g., REITs, equities)
- Compare CVaR optimization against mean-variance and Sharpe ratio portfolios
- Extend model to support constraints (e.g., sector limits, leverage caps)
- Integrate alternative risk measures (e.g., Entropic Risk, Drawdown)

---

## Reference

1. Rockafellar, R.T., & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. *Journal of Risk*, 2(3), 21–41.


