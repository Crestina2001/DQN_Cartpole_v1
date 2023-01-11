from replay import random
import matplotlib.pyplot as plt
from networks import torch
from params import episode_durations, EPS_END, EPS_START, env, \
    device, EPS_DECAY, policy_net, steps_done
import math



def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

def select_action(state):
    global steps_done
    # Return random number between 0.0 and 1.0:
    sample=random.random()
    eps_threshold=EPS_END+(EPS_START-EPS_END)* math.exp(-1.*steps_done/EPS_DECAY)
    steps_done+=1
    if sample>eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1,1)
    else:
        return torch.tensor([[env.action_space.sample()]],device=device,dtype=torch.long)

def select_action_test(state):
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        return policy_net(state).max(1)[1].view(1, 1)