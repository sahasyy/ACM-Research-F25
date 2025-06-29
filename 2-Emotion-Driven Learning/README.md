![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Paper Implementation 2: Emotion-Driven Learning and Evolutionary Game Theory in Multi-Agent Systems with AI/ML Analysis

## Requirements
To install the necessary packages for this project, use the provided requirements.txt file. You can install the dependencies by running:

bash
Copy
Edit
pip install -r requirements.txt
## Project Summary
This repository contains a Python-based simulation inspired by the paper "Understanding Emergent Behaviours in Multi-Agent Systems with Evolutionary Game Theory" by Han et al. The project extends their concepts of cognitive, emotional, and strategic behavior in multi-agent systems by focusing on how beliefs or ideologies spread through social influence, network pressure, and agent-level resistance.

Rather than relying on traditional payoff-driven game theory, this implementation explores belief alignment and ideological polarization through a network of agents who adjust their stance based on neighbor influence, internal stubbornness, and external propagandistic forces.

## Motivation
Throughout history, societal transformations—such as rebellions, colonial resistance, and political radicalization—have been driven not just by material payoff, but by shifts in belief and identity. This simulation aims to capture the emergent dynamics of belief spread in populations, using computational models to explore when societies tip toward cooperation, assimilation, or revolt.

Using AI/ML tools to analyze simulation data allows for predictive modeling and deeper understanding of emergent behavioral patterns, moving from purely descriptive to explanatory and forecast-capable frameworks.

Key Concepts and Contributions
Belief Score Dynamics:
Each agent holds a belief value ranging from -1 (full resistance) to +1 (full assimilation), which evolves based on peer influence and internal resistance (stubbornness).

Influence-Based Updating:
Agents average the beliefs of their neighbors and adjust their stance, modulated by their stubbornness and any external ideological pressure (e.g. state propaganda or rebel messaging).

Network Topology Effects:
Belief propagation is analyzed over various network structures (random, small-world, scale-free) to examine how social connectivity impacts ideological convergence or division.

Emergent Behavior Tracking:
Plots and statistics track how collective belief shifts over time, highlighting phenomena such as tipping points, echo chambers, and belief polarization.

## Implementation Overview
The repository includes:

A minimal agent class with belief values, stubbornness, and influence dynamics

A network-based simulation using networkx

Optional influencer nodes simulating propaganda or charismatic leaders

Visualizations of belief spread and polarization over time

ML-ready data structures for later prediction or forecasting (e.g., which agents radicalize)

## Dataset
The current simulation uses synthetic networks and randomized parameters. Historical or real-world datasets (e.g., population ideology surveys, colonial event timelines, or protest data) could be incorporated to ground the model in empirical evidence.

Usage
Initialize a network of agents with random beliefs and resistance levels.

Simulate multiple time steps of peer-to-peer belief updating.

Inject external influences to simulate top-down control or grassroots resistance.

Visualize collective belief dynamics over time.

## Example Analyses
Belief polarization across network types

Effect of stubbornness on consensus formation

Emergence of ideological clusters

Role of influencers in tipping population alignment

Forecasting belief shift tipping points with ML models

## Citation
**This repository is a paper implementation inspired by:**

Anh Tuan Han et al. (2022), “*Understanding Emergent Behaviours in Multi-Agent Systems with Evolutionary Game Theory,*”
[arXiv:2205.07369](https://arxiv.org/pdf/2205.07369)

Related foundational works on evolutionary game theory, reinforcement learning, and agent-based modeling in social systems.

