import gym
from networks import torch, DQN, optim
from replay import ReplayMemory

# Make the environment
# https://www.gymlibrary.dev/environments/classic_control/cart_pole/
env = gym.make('CartPole-v1')

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
'''
Get the number of state observations
env.reset(): This command will reset the environment. 
It returns an initial observation.
'''
state, _ = env.reset()
n_observations = len(state)

policy_net=DQN(n_observations,n_actions).to(device)
target_net=DQN(n_observations,n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

memory=ReplayMemory(10000)

# to be used in the function select_action(state)
steps_done=0

# to be used in plot_durations(show_result=False) in utils.py
episode_durations = []

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50