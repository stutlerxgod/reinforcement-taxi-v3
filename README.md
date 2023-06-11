# reinforcement-taxi-v3
Q-learning, SARSA, Value-iteration for Taxi-v3 problem of python Gym library.
Reinforcement Learning

## Taxi-v3
The taxi-v3 problem is a classic reinforcement learning problem in the Python library Gym.
The goal is to learn an agent how to navigate a grid-world environment as a taxi driver, picking up passengers and dropping them off at their desired locations.
The environment is represented by a 5×5 grid, with walls blocking certain paths and 
passengers and destinations represented by colored squares. 500 states.
The agent receives a reward for successfully picking up and dropping off passengers. However, the agent also incurs a penalty for each time step is taken and for illegal actions, such as attempting to pick up a passenger who is already in the taxi.
Actions: move south, move north, move east, move west, pick up a passenger, drop off a passenger.
More info here: Taxi - [Gym Documentation](https://www.gymlibrary.dev/)){:target="_blank" rel="noopener"}

