import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy

def exp_avg(data, beta):
    result = np.zeros_like(data)
    for i, datum in enumerate(data):
        if i == 0:
            result[i] = 0
        else:
            result[i] = beta * result[i-1] + (1-beta) * data[i]
            #result[i] = result[i]/(1-beta**i)
            a = 1

    return result

# Open pickled file


infile = open('current_best/score_history_low_gl_sk_2000','rb')
#infile = open('score_history_5000_2.21','rb')
data_dict = pickle.load(infile)
infile.close()

#cleans up the data so a win is counted as 1, draw as zero, loss as -1
win_counter = copy.deepcopy(data_dict['team_0_score'])
win_counter[win_counter > 0] = 1
win_counter[win_counter < 1] = 0
avg_win_percent = exp_avg(win_counter, .98)

non_loss_counter = copy.deepcopy(data_dict['team_0_score'])
non_loss_counter[non_loss_counter >= 0] = 1
non_loss_counter[non_loss_counter < 0] = 0
avg_non_loss_percent = exp_avg(non_loss_counter, .98)

plt.figure(1)
plt.scatter(data_dict['episode_history'], avg_win_percent, s=7, label='Model Winning')
plt.scatter(data_dict['episode_history'], avg_non_loss_percent, s=7, label='Model Not Losing')
plt.plot([0, np.max(data_dict['episode_history'])], [0.33, 0.33],'r--', label='Random Baseline - Winning')
plt.plot([0, np.max(data_dict['episode_history'])], [0.66, 0.66],'r--', label='Random Baseline - Not Losing')
plt.xlabel('Number of Episodes')
plt.ylabel('Running Average Probability')
plt.legend()

#Look at individual brain's losses
goalie_loss = copy.deepcopy(data_dict['goalie_loss'])
avg_goalie_loss = exp_avg(goalie_loss, .98)

striker_loss = copy.deepcopy(data_dict['striker_loss'])
avg_striker_loss = exp_avg(striker_loss, .98)

plt.figure(2)
plt.scatter(data_dict['episode_history'], avg_goalie_loss, s=6, label='Goalie Loss')
plt.scatter(data_dict['episode_history'], avg_striker_loss, s=6, label='Striker Loss')
plt.ylabel('Average Loss')
plt.xlabel('Number of episodes')
plt.legend()
plt.show()

