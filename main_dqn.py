import sys
sys.path.insert(0, "python/")

from unityagents import UnityEnvironment
import numpy as np
import torch
from dqn.dqn_agent_v2 import Agent
from collections import deque
import time

# --------------------- environment configuration ---------------------
env = UnityEnvironment(file_name="Soccer_Env/Soccer_Linux/Soccer.x86", no_graphics=True, seed=1)
#env = UnityEnvironment(file_name="Soccer_Env/Soccer.exe", no_graphics=False, seed=1)

# --------------------- print the brain names ---------------------
print(env.brain_names)

# set the goalie brain
g_brain_name = env.brain_names[0]
g_brain = env.brains[g_brain_name]

# set the striker brain
s_brain_name = env.brain_names[1]
s_brain = env.brains[s_brain_name]

# --------------------- Look at state and action space---------------------
# reset the environment
env_info = env.reset(train_mode=True)

# number of agents 
num_g_agents = len(env_info[g_brain_name].agents)
print('Number of goalie agents:', num_g_agents)
num_s_agents = len(env_info[s_brain_name].agents)
print('Number of striker agents:', num_s_agents)

# number of actions
g_action_size = g_brain.vector_action_space_size
print('Number of goalie actions:', g_action_size)
s_action_size = s_brain.vector_action_space_size
print('Number of striker actions:', s_action_size)

# examine the state space 
g_states = env_info[g_brain_name].vector_observations
g_state_size = g_states.shape[1]
print('There are {} goalie agents. Each receives a state with length: {}'.format(g_states.shape[0], g_state_size))
s_states = env_info[s_brain_name].vector_observations
s_state_size = s_states.shape[1]
print('There are {} striker agents. Each receives a state with length: {}'.format(s_states.shape[0], s_state_size))

# --------------------- Instantiate agents ---------------------
g_agent = Agent(state_size=g_state_size, action_size=g_action_size, seed=0)
s_agent = Agent(state_size=s_state_size, action_size=s_action_size, seed=0)

# --------------------- Additional Reward ---------------------
def ball_reward(state):
    """
    Params
    ======
        state : current state of striker, 3 stacked 112 element vector
            1: ball
            2: opponent's goal
            3: own goal
            4: wall
            5: teammate
            6: opponent
            7: distance
    """
    reward = 0.0
    # Penalize if ball is not in view
    if not any(state[0::8]):
        reward = -0.03
    # Reward for kicking the ball
    else:
        idx = np.where(state[0::8])[0] # check which ray sees the ball
        distance = state[idx*8 + 7] # get the corresponding distance to ball
        if (np.amin(distance) <= 0.03): # Just picking some thresholds for now.
            reward = 0.3

    return reward

# --------------------- DQN trainer ---------------------
def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    g_losses = []
    g_losses_window = deque(maxlen=100)
    s_losses = []
    s_losses_window = deque(maxlen=100)
    for i_episode in range(1, n_episodes+1):
        env_info  = env.reset(train_mode=True)
        score = 0
        ball_reward_val = 0.0
        
        g_states = env_info[g_brain_name].vector_observations       # get initial state (goalies)
        s_states = env_info[s_brain_name].vector_observations       # get initial state (strikers)

        g_scores = np.zeros(num_g_agents)                           # initialize the score (goalies)
        s_scores = np.zeros(num_s_agents)                           # initialize the score (strikers)  
        
        #for t in range(max_t):
        while True:
            action_g_0 = g_agent.act(g_states[0], eps)        # always pick state index 0
            action_s_0 = s_agent.act(s_states[0], eps)  
            
            # Set other team to random
            action_g_1 = np.asarray( [np.random.choice(g_action_size)] ) 
            action_s_1 = np.asarray( [np.random.choice(s_action_size)] )
            # Train simultaneously
            #action_g_1 = g_agent.act(g_states[1], eps)        # always pick state index 1
            #action_s_1 = s_agent.act(s_states[1], eps) 
            
            # Combine actions
            actions_g = np.array( (action_g_0, action_g_1) )                                    
            actions_s = np.array( (action_s_0, action_s_1) )
            actions = dict( zip( [g_brain_name, s_brain_name], [actions_g, actions_s] ) )
            
            env_info = env.step(actions)                                                
            # get next states
            g_next_states = env_info[g_brain_name].vector_observations         
            s_next_states = env_info[s_brain_name].vector_observations
            
            # get reward and update scores
            g_rewards = env_info[g_brain_name].rewards
            s_rewards = env_info[s_brain_name].rewards
            g_scores += g_rewards
            s_scores += s_rewards
            
            ball_reward_val += ball_reward(s_states[0])
            
            # check if episode finished
            done = np.any(env_info[g_brain_name].local_done)
            
            # store experiences
            g_agent.step(g_states[0], action_g_0, g_rewards[0], 
                         g_next_states[0], done)
            s_agent.step(s_states[0], action_s_0, s_rewards[0] + ball_reward(s_states[0]), # adding ball reward
                         s_next_states[0], done)

            if done:
                break
                
            g_states = g_next_states
            s_states = s_next_states
                
        # learn
        goalie_loss = g_agent.learn(g_agent.memory.sample(), 0.99) # discount = 0.99
        striker_loss = s_agent.learn(s_agent.memory.sample(), 0.99) # discount = 0.99 
        
        g_losses.append(goalie_loss.item())
        g_losses_window.append(goalie_loss.item())
        #print(goalie_loss.item())
        s_losses.append(striker_loss.item())
        s_losses_window.append(striker_loss.item())
        
        score = g_scores[0] + s_scores[0]
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}\t Goalie Loss:' \
                  '{:.5f}\t Striker Loss: {:.5f}' \
                  '\t Ball Reward: {:.2f}'.format(i_episode, \
                                                  np.mean(scores_window), \
                                                  np.mean(g_losses_window), \
                                                  np.mean(s_losses_window), \
                                                  ball_reward_val), end="")
        #print(s_states[0][0:56])
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\t Goalie Loss:' \
                  '{:.5f}\t Striker Loss: {:.5f}\n' \
                  '\t Ball Reward: {:.2f}'.format(i_episode, \
                                                  np.mean(scores_window), \
                                                  np.mean(g_losses_window), \
                                                  np.mean(s_losses_window), \
                                                  ball_reward_val))
            torch.save(g_agent.qnetwork_local.state_dict(), 'goalie_dqn_V1.pth')
            torch.save(s_agent.qnetwork_local.state_dict(), 'striker_dqn_V1.pth')
    return scores


n_episodes = 10000
#n_episodes = 1
#max_t = 100000
eps_start = 1.0
eps_end = 0.1
eps_decay = 0.9995

# Pick up where we left off
#GOALIE = './trained_models/goalie_dqn_run1.pth'
#STRIKER = './trained_models/striker_dqn_run1.pth'
#GOALIE = './trained_models/goalie_dqn_ballreward2.pth'
#STRIKER = './trained_models/striker_dqn_ballreward2.pth'
#g_agent.qnetwork_local.load (GOALIE )
#s_agent.qnetwork_local.load( STRIKER )

# Train
#scores = dqn(n_episodes, max_t, eps_start, eps_end, eps_decay)
start = time.time()
scores = dqn(n_episodes, 50000, eps_start, eps_end, eps_decay)
end = time.time()

print('Avg time per 10 episodes: %.1f' %((end-start)/(n_episodes/10)))
env.close()