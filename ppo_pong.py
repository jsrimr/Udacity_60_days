import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.distributions import Normal
from torch.distributions import Categorical
import matplotlib.pyplot as plt

import logging
log_file_name = "ppo_pong.log"
logging.basicConfig(filename=log_file_name, filemode='w', format='%(name)s - %(levelname)s - %(message)s', level = logging.DEBUG)

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

from common.multiprocessing_env import SubprocVecEnv
from pong_util import preprocess_single, preprocess_batch

num_envs = 16
env_name = "Pong-v0"
#Hyper params:
hidden_size      = 32
lr               = 1e-3
num_steps        = 128
mini_batch_size  = 256
ppo_epochs       = 3
threshold_reward = 16
load_weight_n = 305000

def make_env():
    def _thunk():
        env = gym.make(env_name)
        return env

    return _thunk

envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

env = gym.make(env_name)

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=8,
                               stride=4,
                               padding=0)
        nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))
#The second convolution layer takes a 20x20 frame and produces a 9x9 frame
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=4,
                               stride=2,
                               padding=0)
        nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))
#The third convolution layer takes a 9x9 frame and produces a 7x7 frame
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=0)
        nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))
#A fully connected layer takes the flattened frame from thrid convolution layer, and outputs 512 features

        self.lin = nn.Linear(in_features=6 * 6 * 64,
                             out_features=512)
        nn.init.orthogonal_(self.lin.weight, np.sqrt(2))
#A fully connected layer to get logits for ππ
        self.pi_logits = nn.Linear(in_features=512,
                                   out_features=6)
        nn.init.orthogonal_(self.pi_logits.weight, np.sqrt(0.01))  # softmax 없어도 괜찮을까? -> relu 이기 때문에 괜찮다. 음수 안들어간다
#A fully connected layer to get value function
        self.value = nn.Linear(in_features=512,
                               out_features=1)
        nn.init.orthogonal_(self.value.weight, 1)
    
    def forward(self, obs):

        h = F.relu(self.conv1(obs))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.reshape((-1, 6 * 6 * 64))

        h = F.relu(self.lin(h))

        pi = Categorical(logits=self.pi_logits(h))
        value = self.value(h)

        return pi, value


def plot(frame_idx, rewards):
    # clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()
    
def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = preprocess_single(state)
        state = state.expand(1,1,80,80)
        dist, _ = model(state)
        # dist = Categorical(prob)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
        # print(total_reward)
    return total_reward

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage): # 전체 배치에서 mini_batch 를 만드는 것이다.
    batch_size = states.size(0)
    ids = np.random.permutation(batch_size)
    ids = np.split(ids, batch_size // mini_batch_size)
    for i in range(len(ids)):
        yield states[ids[i], :], actions[ids[i], :], log_probs[ids[i], :], returns[ids[i], :], advantage[ids[i], :]
        

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2): # training
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, vlaue = model(state)
            new_log_probs = dist.log_prob(action)
            
            #pi, value = model(state)
            # pi_a = pi.gather(1,action.unsqueeze(-1))
            # logging.warning(f'{pi_a} : pi_a')
            # new_log_probs = torch.log(pi_a)

            ratio = (new_log_probs - old_log_probs).exp()
            # print("ratio :",  ratio)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage


            with torch.autograd.detect_anomaly():
                actor_loss  = - torch.min(surr1, surr2).mean()
                # print("actor loss", actor_loss)
                critic_loss = (return_.detach() - value).pow(2).mean()
                # print("critic loss", critic_loss)
                entropy = dist.entropy()

                loss = 0.5 * critic_loss + actor_loss  - 0.01 * entropy

                optimizer.zero_grad()
                
                loss.sum().backward()
                
                optimizer.step()
        # print(loss.sum())

num_inputs  = envs.observation_space.shape
num_outputs = envs.action_space.n



state = envs.reset()

model = ActorCritic().to(device) #return dist, v
# model.load_state_dict(torch.load(f'weight/pong_{load_weight_n}.pt'))
optimizer = optim.Adam(model.parameters(), lr=lr)

max_frames = 150000
frame_idx  = 0
test_rewards = []
early_stop = False


while not early_stop:
    log_probs = []
    values    = []
    states    = []
    actions   = []
    rewards   = []
    masks     = []
    next_states = []
    
    for _ in range(num_steps): #경험 모으기 - gpu 쓰는구나 . 하지만 여전히 DataParallel 들어갈 여지는 없어보인다. 
        #-> 아.. state 가 벡터 1개가 아닐 것 같다.. 16개네. gpu 쓸만하다. DataParallel 도 가능할듯?
        
        state = preprocess_batch(state)
        
        dist, value = model(state)
        # print(value, value.shape , "value")
        # m = Categorical(dist)
        
        action = dist.sample()
        
        
        next_state, reward, done, _ = envs.step(action.cpu().numpy()) #done 한다고 끝내질 않네??
        
        # logging.warning(f'dist[action] : {dist[action]}')
        # print(action)
        log_prob = dist.log_prob(action) #torch.log(dist[action])
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
        
        states.append(state)
        actions.append(action)
        
        next_states.append(preprocess_batch(next_state))
        
        state = next_state
        frame_idx += 1
        
        
        if frame_idx % 1000 == 0 : # 1000번 마다 plot 그려주기
            print(frame_idx)
            torch.save(model.state_dict(),'weight/pong_{}.pt'.format(frame_idx+load_weight_n))

            test_reward = np.mean([test_env() for _ in range(10)])
            test_rewards.append(test_reward)
            # plot(frame_idx, test_rewards)
            print("test_reward : ", np.mean(test_rewards))
            if test_reward > threshold_reward: early_stop = True

        # stop if any of the trajectories is done
        # we want all the lists to be retangular
        if done.any():
            break
            
    #경험 1세트 모은거로 학습하기
    
    #num_step 만큼 진행했을 때 reward 얼마였는지 구하기
    next_state = preprocess_batch(next_state)
#     print("next_state shape: ", next_state.shape) # [16, 1, 80,80]
    _, next_value = model(next_state)
#     print("next_vlaue shape: " , next_value.shape)
    returns = compute_gae(next_value, rewards, masks, values)
#     logging.debug(f"{returns} and shape is {len(returns)}, {len(returns[0])}" )
    returns = torch.cat(returns).detach()
#     logging.debug("after")
#     logging.debug(f"{returns} and shape is {returns.shape}" )
#     print(returns.shape, "what's happening here?")
    log_probs = torch.cat(log_probs).detach()
    values    = torch.cat(values).detach()
    states    = torch.cat(states)
    actions   = torch.cat(actions)
    
    advantage = returns - values
    
ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)
