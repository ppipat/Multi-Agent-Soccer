# Multi Agent Soccer DRL

Soccer game from Unity Machine Learning Agents Toolkit environment

## Requirements:
- Python 3.6
- tensorflow:  >=1.7 <1.8
- Pillow: >=4.2.1
- matplotlib
- numpy: >=1.13.3,<=1.14.5
- pytest: >=3.2.2,<4.0.0
- docopt
- pyyaml
- protobuf: >=3.6 <3.7
- grpcio: >=1.11.0 <1.12.0
- jupyter

For more information about requirements, follow instructions for https://github.com/Unity-Technologies/ml-agents

Soccer environment for Windows and MacOS can found in 'Soccer_Env/' directory.

## Task
Train soccer team consisting of striker and keeper using Deep Reinforcement Learning.
The goal is to maximize the sum of rewards of teammates.

## Reward Structure
- Keeper
	- -1 when ball enters team's goal
	- +0.1 when ball enters opponent's goal
	- +0.001 existential bonus
- Striker
	- +1 when ball enters opponent's goal
	- -0.1 when ball enters own team's goal
	- -0.001 existential penalty
	
## Baseline
DQN
Hyperparameters:

TODO: add plot of training reward?

## PPO?

## References
- Unity ML-Agents Toolkit https://github.com/Unity-Technologies/ml-agents
- 
