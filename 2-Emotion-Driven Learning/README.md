![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Paper Implementation 2: Emotion-Driven Learning and Evolutionary Game Theory in Multi-Agent Systems with AI/ML Analysis

## Requirements
To install the necessary packages for this project, use the provided requirements.txt file. You can install the dependencies by running:

bash
Copy
Edit
pip install -r requirements.txt
## Project Summary
This repository contains a Python-based simulation that extends classical Evolutionary Game Theory (EGT) and Agent-Based Modeling (ABM) by incorporating cognitive and emotional mechanisms within agents who adapt their strategies via reinforcement learning. The simulation explores how internal states like trust, guilt, and memory influence the emergence and stability of prosocial behaviors in complex social networks.

Additionally, the project integrates machine learning techniques—including random forest classifiers and time series forecasting—to analyze, predict, and explain the dynamics of strategy adoption and cooperation over time.

## Motivation
Traditional evolutionary game models often assume that agents update strategies purely based on payoffs and simple imitation. However, real-world social behavior is influenced by cognitive biases and emotional factors, which affect decision-making and learning. This implementation bridges that gap by embedding emotion-aware learning agents within evolutionary games, providing richer insight into the mechanisms driving cooperation and conflict in multi-agent systems.

Using AI/ML tools to analyze simulation data allows for predictive modeling and deeper understanding of emergent behavioral patterns, moving from purely descriptive to explanatory and forecast-capable frameworks.

Key Concepts and Contributions
Emotion-Driven Reinforcement Learning: Agents adapt strategies not only based on payoffs but also emotional states (e.g., guilt reduces exploitative actions), shaping decision policies.

Heterogeneous Agent Memory: Agents maintain histories of interactions, influencing future choices and emotional states.

Network Effects on Emergence: Experiments conducted on various network topologies (lattice, scale-free, random graphs) reveal how structure impacts cooperation.

AI/ML Integration: Random forest models predict agent strategy changes based on internal and external states, while forecasting algorithms model global cooperation trends and detect early warning signals.

Behavioral Pattern Discovery: Clustering and dimensionality reduction techniques expose latent strategy groups and phase transitions in the multi-agent system.

## Implementation Overview
The repository includes:

An ABM framework (e.g., Mesa) implementing agents with internal cognitive and emotional states.

Reinforcement learning algorithms (Q-learning or policy gradients) with emotion-weighted reward functions.

Functions for generating social networks and initializing heterogeneous agent populations.

Modules for collecting and preprocessing simulation data for ML analysis.

Scripts applying Random Forest classification to predict strategy updates.

Time series forecasting tools (e.g., ARIMA, LSTM) for global cooperation trend prediction.

Visualization tools for emergent behaviors, feature importances, and forecast accuracy.

## Dataset
The simulations generate synthetic time-series and networked agent data. For real-world application, behavioral indices (e.g., social trust, inequality), economic indicators, or historical data can parameterize agent traits and inform initial conditions.

Usage
Configure agent parameters (emotional sensitivity, memory length), network type, and game payoff matrix.

Run evolutionary game simulations with learning-enabled agents over multiple iterations.

Extract and preprocess agent-level and system-level data.

Train Random Forest models to predict agent behavior switches.

Apply forecasting algorithms to cooperation metrics for trend analysis.

Visualize emergent cooperation patterns, behavioral clusters, and predictive insights.

## Example Analyses
Cooperation rate evolution across different network structures.

Impact of emotional reward modifiers on strategy stability.

Feature importance ranking for predictors of agent strategy shifts.

Forecasted vs actual cooperation trajectories and early rebellion detection.

Behavioral cluster maps and transition dynamics.

## Citation
**This repository is a paper implementation inspired by:**

Anh Tuan Han et al. (2022), “Understanding Emergent Behaviours in Multi-Agent Systems with Evolutionary Game Theory,”
[arXiv:2205.07369](https://arxiv.org/pdf/2205.07369)

Related foundational works on evolutionary game theory, reinforcement learning, and agent-based modeling in social systems.

