import gymnasium as gym
import tqdm
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import ReplayMemory, Transition


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
    
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, n_hidden=128):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_hidden)
        self.layer3 = nn.Linear(n_hidden, n_actions)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
class DQNAgent(nn.Module):
    def __init__(self, env, batch_size, gamma, eps_start, eps_end, eps_decay, tau, lr, device, memory_size):
        super().__init__()
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.device = device
        
        n_actions = env.action_space.n
        state, _ = env.reset()
        n_observations = len(state)

        
        self.policy_net = DQN(n_observations=n_observations, n_actions=n_actions).to(device)
        self.target_net = DQN(n_observations=n_observations, n_actions=n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        
        self.memory = ReplayMemory(memory_size)
        self.steps_done = 0
        self.loss = 0
        self.episode_durations = []
        self.loss_history = []
    
    def select_action(self, state):
        sample=random.random()
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps:
            return self.policy_net(state).argmax(-1).view(1, -1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long) ## 
            
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch=Transition(*zip(*transitions))
        
        non_final_mask=torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        self.state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        self.next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            self.next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        self.expected_state_action_values = (self.next_state_values*self.gamma) + reward_batch
        
        criterion = nn.SmoothL1Loss()
        self.loss = criterion(self.state_action_values, self.expected_state_action_values.unsqueeze(1)) ##
        self.loss_history.append(self.loss)
        self.optimizer.zero_grad()
        self.loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
    def train(self, episodes):
        try: 
            assert torch.cuda.is_available()
            num_episodes = episodes
        except:
            print("CUDA Unavailable")
            num_episodes = 50
            
        for _ in tqdm.tqdm(range(num_episodes), ncols=100):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                self.memory.push(state, action, next_state, reward)
                state = next_state
                self.optimize_model()
                
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1 - self.tau)
                self.target_net.load_state_dict(target_net_state_dict)
                
                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_duration()
                    break
                
    def savefig(self, root=None):
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
        plt.title('Result')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.savefig(root if root is None else 'dqn.png')
        
            
    def plot_duration(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title("Result")
        else:
            plt.clf()
            plt.title("Training")
            plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)  # pause a bit so that plots are updated
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
                
if __name__ == '__main__':
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    LR = 1e-4
    TAU=0.005
    EPOCHS = 700

    env = gym.make('CartPole-v1')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(env=env, batch_size=BATCH_SIZE, gamma=GAMMA, eps_start=EPS_START, eps_end=EPS_END, 
                 eps_decay=EPS_DECAY, tau=TAU, lr=LR, device=device, memory_size=10000)
    agent.train(EPOCHS)
    agent.savefig()