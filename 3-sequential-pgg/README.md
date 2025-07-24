![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Paper Implementation 2: Emotion-Driven Learning and Evolutionary Game Theory in Multi-Agent Systems with AI/ML Analysis

## Requirements
To install the necessary packages for this project, use the provided requirements.txt file. You can install the dependencies by running:

bash
Copy
Edit
pip install -r requirements.txt

# Sequential Public Goods Game  
**Learning Cooperation and Strategy in Temporally Extended Social Dilemmas**

This repository explores the intersection of reinforcement learning and evolutionary game theory through a sequential formulation of the classic Public Goods Game (PGG). Inspired by Leibo et al.'s *Sequential Social Dilemmas*, this project investigates how cooperation emerges as a property of learned policies rather than atomic actions.

## Motivation  
Traditional matrix games like the Public Goods Game and Prisoner’s Dilemma reduce social dilemmas to single-shot decisions: cooperate or defect. However, real-world cooperation is temporally extended, context-sensitive, and policy-driven. This project reframes the PGG as a **Markov Game**, where agents must learn temporally consistent cooperative behaviors in a dynamic environment.

We aim to understand how factors such as agent incentives, spatial dynamics, and environmental scarcity affect the emergence and sustainability of cooperation.

---

## Environment  
The environment is a **gridworld** in which multiple agents interact, contribute to a shared public good, and receive rewards based on joint behavior.

### Features:
- **Agents**: Multiple independent agents with partial observability.
- **State**: Spatial layout, agent positions, and public resource indicators.
- **Actions**:
  - Move (Up, Down, Left, Right, Stay)
  - Contribute to local public pool (at a cost)
  - Withhold contribution (defect)
- **Payoffs**: Total contributions are multiplied by a synergy factor and shared among neighbors.
- **Temporal Dynamics**: Agents must learn when and where to contribute based on evolving context.

---

## Algorithms  
We use **independent reinforcement learning agents** to study multi-agent adaptation.

### Supported Methods:
- **Independent Deep Q-Networks (IDQN)**
- **Policy Gradient Variants (e.g., PPO, A2C)**
- (Optional) **Evolutionary Selection** of policies across generations

Each agent optimizes its own policy independently, capturing decentralized adaptation and allowing for emergent strategic diversity.

---

## Experiments  
We analyze how cooperation dynamics shift across environments:

### Key Experimental Axes:
- **Resource Abundance**: Sparse vs. abundant rewards
- **Population Density**: How crowdedness affects cooperation
- **Observation Radius**: Influence of local vs. global awareness
- **Cost of Contribution**: Incentives for free-riding

### Metrics:
- Average contribution rates  
- Reward inequality  
- Temporal consistency of cooperative behavior  
- Cluster emergence (spatial or temporal)

---

## Results (In Progress)  
Preliminary simulations reveal:
- Cooperation is harder to sustain under high cost or low visibility.
- Agents learn opportunistic strategies (e.g., contribute only when others do).
- Environmental pressure (e.g., scarcity) can catalyze cooperation or conflict.

Final results will include visualizations of agent behavior, payoff trajectories, and cooperation heatmaps.

---

## Future Work  
- **Punishment mechanisms** (e.g., tag-and-penalize defectors)
- **Heterogeneous agent types** (mix of selfish, altruistic, learning)
- **Evolutionary adaptation** (fitness-based selection over generations)
- **Transfer learning** across dilemma types (PGG → PD)

---

## References  
- Leibo, J. Z., Hughes, E., Lanctot, M., Zambaldi, V., & Graepel, T. (2017). [Multi-agent reinforcement learning in sequential social dilemmas](https://arxiv.org/abs/1702.03037).
- Hardin, G. (1968). The tragedy of the commons. *Science*.
- Perc, M. et al. (2017). Statistical physics of human cooperation. *Physics Reports*.
