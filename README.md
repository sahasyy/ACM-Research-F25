# Maximum Entropy Portfolio Optimization

This Jupyter Notebook implements a simplified version of the **Maximum Entropy Approach to Portfolio Optimization**, based on the research paper:

"Maximum Entropy Approach to Portfolio Optimization: Economic Justification of an Intuitive Diversity Idea"  
by Laxman Bokati & Vladik Kreinovich (2019)

---

## Objective

In classical finance, Markowitz-style portfolio optimization depends on accurate estimates of both mean returns and covariances. However, in many real-world situations, investors only have access to expected returns — not covariances or volatility estimates. This creates a challenge: how can a rational, diversified portfolio be constructed with limited statistical information?

This research uses Shannon entropy from information theory as a diversification metric and proposes maximizing entropy as a logical method for portfolio construction under uncertainty. The maximum entropy portfolio corresponds to the most unbiased weight distribution given only partial knowledge.

---

## What This Notebook Demonstrates

This notebook:

1. Simulates a set of assets with known expected returns but unknown covariances.
2. Computes the maximum entropy portfolio, which is a uniform allocation across all assets.
3. Generates random portfolios using Dirichlet distributions to simulate alternative allocations.
4. Calculates the entropy and expected return of each random portfolio.
5. Visualizes the tradeoff between entropy and expected return, demonstrating the diversification benefit of entropy maximization.

---

## Technical Concepts

- **Shannon Entropy**  
  Used to measure portfolio diversity:
  \[
  H(w) = -\sum_i w_i \log(w_i)
  \]
  where \( w_i \) are the portfolio weights.

- **Maximum Entropy Principle**  
  Selects the most diversified portfolio consistent with known constraints (in this case, only the expected returns).

- **Dirichlet Sampling**  
  Generates thousands of valid portfolios whose weights sum to 1 and are non-negative.

- **Expected Return**  
  Calculated using the dot product of portfolio weights and the expected return vector.

---

