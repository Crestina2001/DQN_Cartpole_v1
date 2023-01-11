from networks import torch, nn
from params import episode_durations, policy_net,target_net, memory, BATCH_SIZE, device, GAMMA, optimizer, \
    num_episodes, gym, env, TAU
from replay import Transition
from itertools import count
from utils import select_action,select_action_test, plot_durations, plt

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    # a list of length 128
    # each element looks like Transition(state=tensor([[ 0.0787,  0.1479, -0.0796, -0.2615]], device='cuda:0')
    # action=tensor([[1]], device='cuda:0'),
    # next_state=tensor([[ 0.0817,  0.3440, -0.0848, -0.5782]], device='cuda:0'),
    # reward=tensor([1.], device='cuda:0'))
    transitions = memory.sample(BATCH_SIZE)

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    # This is a transition, which look like Transition(state=(),action=(),next_state=(),reward=())
    # Take batch.state for example, this is a tuple of length 128
    # Each element of the tuple looks like tensor([[ 0.0513,  0.9795, -0.0543, -1.3245]], device='cuda:0')
    batch = Transition(*zip(*transitions))

    # mask tutorial: https://towardsdatascience.com/the-concept-of-masks-in-python-50fd65e64707
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    # non_final_mask is a torch.tensor of length 128, whose value is True or False,
    # denoting whether each element in batch.next_state is None or not
    # (True for not None, and False for None)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    # concatenate all not None batch.next_state values
    # torch.cat: https://pytorch.org/docs/stable/generated/torch.cat.html
    # len(non_final_next_states)=121,123,etc.
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    # shape of state_batch: torch.Size([128, 4])
    state_batch = torch.cat(batch.state)
    # shape of state_batch: torch.Size([128, 1])
    action_batch = torch.cat(batch.action)
    # shape of state_batch: torch.Size([128, 1])
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # shape of policy_net(state_batch): torch.Size([128, 2])
    # shape of state_action_values: torch.Size([128, 1])
    # select the policy_net value of the action.
    # e.g- policy_net: [0.123,0.982], action_batch: [0], state_action_values: [0.123]
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    # shape of next_state_values: torch.Size([128])
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        # It is where the network is learning: if the game is terminated, then
        # next_state_values will be 0
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    # reward_batch is all 1(128 1s)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    #print(loss)
    #print(state_action_values[0])
    #print(expected_state_action_values.unsqueeze(1)[0])

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def train():
    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        if gym.__version__[:4] == '0.26':
            state, _ = env.reset()
        elif gym.__version__[:4] == '0.25':
            state, _ = env.reset(return_info=True)
        # state looks like: tensor([[-0.0256,  0.0395, -0.0394, -0.0362]], device='cuda:0')
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        if (i_episode + 1) % 100 == 0:
            PATH = 'models/checkpoint' + str(i_episode + 1) + '.pt'
            torch.save(policy_net.state_dict(), PATH)

        # itertools.count(start=0, step=1)
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break
    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()

def test(num=3,PATH='models/checkpoint600.pt'):
    global policy_net
    policy_net.load_state_dict(torch.load(PATH))
    policy_net.eval()
    # some error may happen if your gym version is not 0.26
    # and you may need to install pygame manually
    env=gym.make('CartPole-v1',render_mode="human")
    for i in range(num):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action_test(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Move to the next state
            state = next_state

            if done:
                print(t+1)
                break