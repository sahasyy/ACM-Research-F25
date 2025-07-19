# CVaR-Optimized Portfolio with Simulated Returns

This Jupyter Notebook implements a simple Conditional Value-at-Risk (CVaR) portfolio optimization using only NumPy and matplotlib, without relying on external optimization libraries like `cvxpy`.

## Objective

To find the optimal portfolio weights across a set of simulated assets that minimizes CVaR at a 95% confidence level. This provides a practical example of risk-aware asset allocation using simulated return distributions.

## Methodology

### 1. Data Simulation

- Simulate daily returns for 4 assets over 252 trading days (1 year).
- The asset returns are generated using a multivariate normal distribution with a specified mean vector and covariance matrix.

### 2. CVaR Computation

- CVaR (Conditional Value-at-Risk) at confidence level alpha is the expected portfolio loss assuming the loss exceeds the Value-at-Risk (VaR) at that level.

### 3. Optimization via Random Sampling

- A brute-force method is used to approximate the optimal portfolio weights.
- Random weight vectors that sum to 1 are generated using the Dirichlet distribution.
- For each weight vector, the portfolio CVaR is computed.
- The weight vector that yields the minimum CVaR is recorded as the optimal allocation.

### 4. Visualization

- A histogram of the optimized portfolio's return distribution is plotted.
- The Value-at-Risk (VaR) threshold is marked to illustrate where CVaR begins.
