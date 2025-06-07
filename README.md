![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Summer 2025 Paper Implementations

## Requirements

To install the necessary packages for this project, use the provided requirements.txt file. You can install the dependencies by running:

`pip install -r requirements.txt`

## Project Summary

This repository contains a Python-based simulation of the contagion dynamics described in the paper "Long ties accelerate noisy threshold-based contagions". The paper investigates how social behaviors spread through a population when adoption depends on local peer influence. It focuses on threshold-based contagion models and evaluates how adding long-range connections and decision noise affects the speed and extent of spread.

## Motivation

In many social systems, individuals adopt behaviors only after multiple peers have done so. Traditional models emphasize accuracy or final adoption levels but often ignore how quickly these behaviors spread. This paper shifts focus toward speed and structure. It examines whether adding random, long-distance connections to clustered networks can lead to faster adoption, especially when people sometimes adopt even when their threshold is not met.

## Key Concepts and Contributions

**Noisy Threshold Adoption:** Nodes adopt behaviors based on a threshold number of neighbors who have adopted, with a small probability of adopting below the threshold to simulate noise or randomness in behavior.

- Impact of Long Ties: Adding random long-range edges to clustered networks significantly speeds up contagion under noisy conditions.

- Behavioral Reversibility: The model can include scenarios where nodes reverse adoption decisions, adding realism to the simulation.

- Empirical Simulations: Simulations show that adding long ties generally reduces the number of steps required for widespread adoption.

## Implementation Overview

**The repository includes:**

Functions for creating synthetic social networks using ring lattices and random rewiring.

A configurable contagion simulator based on individual thresholds and noise.

Tools for comparing spread dynamics across different network structures.

## Dataset

The simulations in this project use synthetic networks. For experiments on real-world data, datasets such as Facebook ego networks or other social graphs can be incorporated with minor adjustments.

## Usage

Generate a base network (e.g., a ring lattice).

Optionally rewire a fraction of edges to create long ties.

Run the contagion simulation using a threshold and a noise value.

Plot the number of adopters over time and compare spread rates.

## Example Analyses

Adoption curve over time steps

Effect of noise on diffusion rate

Comparison between networks with and without long ties

## Citation

**This repository is a paper implementation of:**
Dean Eckles, Elchanan Mossel, M. Amin Rahimian, Subhabrata Sen
"_Long ties accelerate noisy threshold-based contagions._"
[arXiv:1810.03579](https://arxiv.org/pdf/1810.03579v5)

**For the authors' original codebase and simulation tools, see:**
[social-contagion (Amin Rahimian)](https://github.com/aminrahimian/social-contagion)
