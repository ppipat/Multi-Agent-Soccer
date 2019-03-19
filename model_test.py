import sys
sys.path.insert(0, "python/")

from unityagents import UnityEnvironment
import numpy as np
import torch
from dqn.dqn_agent_v2 import Agent
from collections import deque

from model import ActorModel, CriticModel
from agent import Agent as PpoAgent

env = UnityEnvironment(file_name="Soccer_Env/Soccer.app", no_graphics=True, seed=1)
#env = UnityEnvironment(file_name="Soccer_Env/Soccer_Linux/Soccer.x86", no_graphics=True, seed=1)

# print the brain names
print(env.brain_names)

# set the goalie brain
g_brain_name = env.brain_names[0]
g_brain = env.brains[g_brain_name]

# set the striker brain
s_brain_name = env.brain_names[1]
s_brain = env.brains[s_brain_name]


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


# Load trained agents and run
# DQN
# g_agent_red = Agent(state_size=g_state_size, action_size=g_action_size, seed=0)
# s_agent_red = Agent(state_size=s_state_size, action_size=s_action_size, seed=0)
g_agent_blue = Agent(state_size=g_state_size, action_size=g_action_size, seed=0)
s_agent_blue = Agent(state_size=s_state_size, action_size=s_action_size, seed=0)
# PPO
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
goalie_actor_model = ActorModel( g_state_size, g_action_size ).to(DEVICE)
striker_actor_model = ActorModel( s_state_size, s_action_size ).to(DEVICE)
N_STEP = 8



# RED TEAM -------------------------------------
# DQN_base ----
GOALIE_red = './trained_models/goalie_dqn_V2_mod.pth'
STRIKER_red = './trained_models/striker_dqn_V2_mod.pth'
# DQN_1 ----
# GOALIE_red = './trained_models/goalie_dqn_V1.pth'
# STRIKER_red = './trained_models/striker_dqn_V1.pth'
# DQN_1_mod ----
# GOALIE_red = './trained_models/goalie_dqn_V1_modified.pth'
# STRIKER_red = './trained_models/striker_dqn_V1_modified.pth'
# DQN_2 ----
# GOALIE_red = './trained_models/goalie_dqn_V2.pth'
# STRIKER_red = './trained_models/striker_dqn_V2.pth'
# DQN_2_mod ----
# GOALIE_red = './trained_models/goalie_dqn_V2_mod.pth'
# STRIKER_red = './trained_models/striker_dqn_V2_mod.pth'

# g_agent_red.qnetwork_local.load (GOALIE_red )
# s_agent_red.qnetwork_local.load( STRIKER_red )
# PPO_1 ----
# goalie_actor_model.load( './trained_models/checkpoint_goalie_actor_v1.pth' )
# striker_actor_model.load( './trained_models/checkpoint_striker_actor_v1.pth' )
g_agent_red = PpoAgent( DEVICE, 0, goalie_actor_model, N_STEP )
s_agent_red = PpoAgent( DEVICE, 0, striker_actor_model, N_STEP )

# BLUE TEAM -------------------------------------
# DQN_base
# GOALIE_blue = './trained_models/goalie_dqn_run1.pth'
# STRIKER_blue = './trained_models/striker_dqn_run1.pth'
# DQN_1 ----
# GOALIE_blue = './trained_models/goalie_dqn_V1.pth'
# STRIKER_blue = './trained_models/striker_dqn_V1.pth'
# DQN_1_mod ----
# GOALIE_blue = './trained_models/goalie_dqn_V1_modified.pth'
# STRIKER_blue = './trained_models/striker_dqn_V1_modified.pth'
# DQN_2 ----
# GOALIE_blue = './trained_models/goalie_dqn_V2.pth'
# STRIKER_blue = './trained_models/striker_dqn_V2.pth'
# DQN_2_mod ----
GOALIE_blue = './trained_models/goalie_dqn_V1.pth'
STRIKER_blue = './trained_models/striker_dqn_V1.pth'

g_agent_blue.qnetwork_local.load (GOALIE_blue )
s_agent_blue.qnetwork_local.load( STRIKER_blue )


team_red_window_score = []
team_red_window_score_wins = []

team_blue_window_score = []
team_blue_window_score_wins = []

draws = []

n_episodes = 500
for i in range(n_episodes):                                       # play game for 2 episodes
    env_info = env.reset(train_mode=True)                  # reset the environment    
    g_states = env_info[g_brain_name].vector_observations  # get initial state (goalies)
    s_states = env_info[s_brain_name].vector_observations  # get initial state (strikers)
    g_scores = np.zeros(num_g_agents)                      # initialize the score (goalies)
    s_scores = np.zeros(num_s_agents)                      # initialize the score (strikers)
    while True:
        # RED TEAM actions
#         action_g_0 = g_agent_red.act(g_states[0], 0)       # always pick state index 0 for red
#         action_s_0 = s_agent_red.act(s_states[0], 0)  
        # Get action for PPO agent
        action_g_0, _ = g_agent_red.act( g_states[0] )
        action_s_0, _ = s_agent_red.act( s_states[0] )

        # BLUE TEAM actions
        # ----- RANDOM -----
        action_g_1 = np.asarray( [np.random.choice(g_action_size)] ) 
        action_s_1 = np.asarray( [np.random.choice(s_action_size)] )
        # ----- Trained -----
#         action_g_1 = g_agent_blue.act(g_states[1], 0)      # always pick state index 1 for blue
#         action_s_1 = s_agent_blue.act(s_states[1], 0)

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
        
        # check if episode finished
        done = np.any(env_info[g_brain_name].local_done)  
        
        # roll over states to next time step
        g_states = g_next_states
        s_states = s_next_states
        
        # exit loop if episode finished
        if done:                                           
            break
    team_red_score = g_scores[0] + s_scores[0]
    team_red_window_score.append( team_red_score )
    team_red_window_score_wins.append( 1 if team_red_score > 0 else 0)        

    team_blue_score = g_scores[1] + s_scores[1]
    team_blue_window_score.append( team_blue_score )
    team_blue_window_score_wins.append( 1 if team_blue_score > 0 else 0 )

    draws.append( team_red_score == team_blue_score )
    print('Scores from episode {}: {} (goalies), {} (strikers)'.format(i+1, g_scores, s_scores))

print('Red Wins: \t{} \tScore: \t{:.5f} \tAvg: \t{:.2f} \tDraws: \t{}'.format( \
                  np.count_nonzero(team_red_window_score_wins)/n_episodes, team_red_score, \
                  np.sum(team_red_window_score), np.count_nonzero(draws)/n_episodes ))

env.close()