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
	
## Baseline - Proximal Policy Optimization (PPO)
PPO model and trainer has been implemented by Marcello Borges https://github.com/marcelloaborges/Soccer-PPO, in which we are using as a baseline model. The code is incorporated under PPO branch of this repository, where the agents can be trained by running main.py. 

## DQN
The Deep-Q Network model has been implemented inside 'dqn/' directory, leveraging the model from Udacity DRL course. The function to train the agents currently lives inside 'Soccer.ipynb' notebook.

TODO: 
- Hyperparameter search
- Adjust network size
- Experience Replay
- Performance Evaluation

## Dueling DQN
TODO:
- implement model
- Train and evaluate

## References
- Unity ML-Agents Toolkit https://github.com/Unity-Technologies/ml-agents
- Udacity Deep Reinforcement Learning https://github.com/udacity/deep-reinforcement-learning
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.
- Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). Prioritized experience replay. arXiv preprint arXiv:1511.05952.
- 
