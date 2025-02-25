import os, pickle, argparse
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from torch.distributions import Categorical
from sparse_init import sparse_init

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        #Removing sparse initialization
        nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
        m.bias.data.fill_(0.0)

class Actor(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128):
        super(Actor, self).__init__()
        self.fc_layer   = nn.Linear(n_obs, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.fc_pi = nn.Linear(hidden_size, n_actions)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.fc_layer(x)
        x = F.leaky_relu(x)
        x = self.hidden_layer(x) 
        x = F.leaky_relu(x)
        pref = self.fc_pi(x)
        return pref

class Critic(nn.Module):
    def __init__(self, n_obs=11, hidden_size=128):
        super(Critic, self).__init__()
        self.fc_layer   = nn.Linear(n_obs, hidden_size)
        self.hidden_layer  = nn.Linear(hidden_size, hidden_size)
        self.linear_layer  = nn.Linear(hidden_size, 1)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.fc_layer(x) 
        x = F.leaky_relu(x)
        x = self.hidden_layer(x) 
        x = F.leaky_relu(x)
        return self.linear_layer(x)

class ClassicAC(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=32, lr=1e-3, gamma=0.99):
        super(ClassicAC, self).__init__()
        self.gamma = gamma
        self.policy_net = Actor(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size)
        self.value_net = Critic(n_obs=n_obs, hidden_size=hidden_size)

        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=lr)

    def pi(self, x):
        preferences = self.policy_net(x)
        probs = F.softmax(preferences, dim=-1)
        return probs

    def v(self, x):
        return self.value_net(x)

    def sample_action(self, s):
        x = torch.from_numpy(s).float()
        probs = self.pi(x)
        dist = Categorical(probs)
        return dist.sample().numpy()

    def update_params(self, s, a, r, s_prime, done):
        done_mask = 0 if done else 1
        s, a, r, s_prime, done_mask = torch.tensor(np.array(s), dtype=torch.float), torch.tensor(np.array(a)), \
                                         torch.tensor(np.array(r)), torch.tensor(np.array(s_prime), dtype=torch.float), \
                                         torch.tensor(np.array(done_mask), dtype=torch.float)

        v_s,v_prime = self.v(s),self.v(s_prime)

        td_target = r + self.gamma * v_prime * done_mask
        delta = td_target - v_s

        probs = self.pi(s)
        dist = Categorical(probs)
        log_prob_pi = dist.log_prob(a)

        actor_loss = -log_prob_pi * delta.detach()

        self.optimizer_policy.zero_grad()
        actor_loss.backward()
        self.optimizer_policy.step()
        critic_loss = (v_s - td_target).pow(2)

        self.optimizer_value.zero_grad()
        critic_loss.backward()
        self.optimizer_value.step()

def main(env_name, seed, lr, gamma, total_steps, debug, render=False):
    torch.manual_seed(seed); np.random.seed(seed)
    env = gym.make(env_name, render_mode='human', max_episode_steps=500) if render else gym.make(env_name, max_episode_steps=500) #Para CartPole max_episode_steps = 500. Antes estaba en 10000
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    agent = ClassicAC(n_obs=env.observation_space.shape[0], n_actions=env.action_space.n, lr=lr, gamma=gamma)
    if debug:
        print("seed: {}".format(seed), "env: {}".format(env.spec.id))
    returns, term_time_steps = [], []
    s, _ = env.reset(seed=seed)
    for t in range(1, total_steps+1):
        a = agent.sample_action(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        agent.update_params(s, a, r, s_prime, terminated or truncated)
        s = s_prime
        if terminated or truncated:
            if debug:
                print("Episodic Return: {}, Time Step {}".format(info['episode']['r'][0], t))
            returns.append(info['episode']['r'][0])
            term_time_steps.append(t)
            terminated, truncated = False, False
            s, _ = env.reset()
    env.close()
    save_dir = "data_classic_ac_{}_lr{}_gamma{}".format(env.spec.id, lr, gamma)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "seed_{}.pkl".format(seed)), "wb") as f:
        pickle.dump((returns, term_time_steps, env_name), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classic AC(λ)')
    parser.add_argument('--env_name', type=str, default='CartPole-v1')
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--total_steps', type=int, default=500_000)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    main(args.env_name, args.seed, args.lr, args.gamma, args.total_steps, args.debug, args.render)