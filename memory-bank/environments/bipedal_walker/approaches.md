# Approaches: Bipedal Walker

This document details the different solution approaches implemented or explored for the Bipedal Walker environment.

## Approach 1: A3C (Asynchronous Advantage Actor-Critic)

- **Source File:** `bipedal_walker-a3c.py`
- **Algorithm:** Asynchronous Advantage Actor-Critic (A3C). This is a policy gradient algorithm that uses multiple parallel agents to explore the environment and update a global network.
- **Key Hyperparameters & Configuration:**
    - (To be filled in by reviewing `bipedal_walker-a3c.py` - e.g., learning rates, discount factor, number of workers, network architecture).
- **Results & Performance:**
    - (To be filled in - e.g., average score achieved, training time, stability of learning).
- **Learnings & Insights:**
    - (To be filled in - e.g., effectiveness of A3C for this problem, challenges in tuning, specific observations about agent behavior).

## Approach 2: Genetic Algorithm

- **Source File:** `bipedal_walker_plus_genetic_algorithm.py`
- **Algorithm:** A genetic algorithm (GA). This is an evolutionary algorithm that maintains a population of solutions (e.g., neural network weights) and iteratively improves them through selection, crossover, and mutation.
- **Key Hyperparameters & Configuration:**
    - (To be filled in by reviewing `bipedal_walker_plus_genetic_algorithm.py` - e.g., population size, mutation rate, crossover strategy, fitness function details).
- **Results & Performance:**
    - (To be filled in - e.g., best fitness achieved, how many generations to converge, robustness of the solution).
- **Learnings & Insights:**
    - (To be filled in - e.g., comparison to RL methods, suitability of GAs for continuous control, challenges in designing the fitness function or genetic operators).

## Approach 3: (Generic/Initial)

- **Source File:** `bipedal_walker.py`
- **Algorithm:** (To be determined by reviewing `bipedal_walker.py`. This might be a simpler RL algorithm, a baseline, or an earlier attempt).
- **Key Hyperparameters & Configuration:**
    - (To be filled in).
- **Results & Performance:**
    - (To be filled in).
- **Learnings & Insights:**
    - (To be filled in).

*(This file will be updated as more details about each approach are documented or as new approaches are developed.)*
