# Copyright 2016 Google Inc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""Basic random agent for DeepMind Lab."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
import os
import deepmind_lab
import imageio
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.autograd as autograd
from torch.autograd import Variable

def _action(*entries):
  return np.array(entries, dtype=np.intc)

class PolicyNet(nn.Module):
  def __init__(self, width, height, actions):
    super(PolicyNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 8, kernel_size=5)
    self.conv1.weight.data.normal_(0, 0.01)
    self.conv1.bias.data.fill_(0)

    self.conv2 = nn.Conv2d(8, 16, kernel_size=5)
    self.conv2.weight.data.normal_(0, 0.01)
    self.conv2.bias.data.fill_(0)

    self.conv3 = nn.Conv2d(16, 16, kernel_size=5)        
    self.conv3.weight.data.normal_(0, 0.01)
    self.conv3.bias.data.fill_(0)

    self.fc1 = nn.Linear(16 * 2 * 2, 16)
    self.fc1.weight.data.normal_(0, 0.01)
    self.fc1.bias.data.fill_(0)

    self.fc2 = nn.Linear(16, actions)
    self.fc2.weight.data.normal_(0, 0.01)
    self.fc2.bias.data.fill_(0)

    self.bn1 = nn.BatchNorm2d(8)
    self.bn2 = nn.BatchNorm2d(16)
    self.bn3 = nn.BatchNorm2d(16)

    # self.fc2 = nn.Linear(64, actions)
    self.gamma = 0.95
    self.reward_seq = []
    self.action_prob_seq = []
    self.action_seq = []

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    # x = self.bn1(x)

    x = F.relu(F.max_pool2d(self.conv2(x), 4))
    # x = self.bn2(x)

    x = F.relu(F.max_pool2d(self.conv3(x), 4))
    # x = self.bn3(x)

    x = x.view(-1, 16 * 2 * 2)
    # x = F.relu(self.fc1(x))        
    # x = F.dropout(x, training=self.training)
    x = F.relu(self.fc1(x))

    x = self.fc2(x)

    return F.softmax(x)

# define memory class for storing short-term memory
class FoodMem():
  def __init__(self, mem_size, mem_dim):
    self.root = '/home/jwyang/Researches/lab/RLCourseProject/'    
    self.mem_size = mem_size
    self.mem_dim = mem_dim
    self.mem_backgrounds = torch.Tensor(mem_size, mem_dim)
    self.mem_apples = torch.Tensor(mem_size, mem_dim)
    self.mem_lemons = torch.Tensor(mem_size, mem_dim)

  def loadmem(self):
    # load memory from walls #
    # get file list of background images
    list = os.listdir(self.root + 'food_memory/data/256background')
    # randomly choose #mem_size images from the list
    rand_id = torch.randperm(len(list))

    for i in xrange(self.mem_size):
      img_path = list[rand_id[i]]
      img = imread(self.root + 'food_memory/data/256background' + '/' + img_path)
      img_c = img[128 - 48:128 + 48, 128 - 48:128 + 48, :]      
      img_r = imresize(img_c, [64, 64])    
      img_r = np.transpose(img_r / 255, (2, 0, 1))
      img_tensor = torch.from_numpy(img_r).float().view(1, mem_dim)
      self.mem_backgrounds.index_copy_(0, torch.LongTensor([i]), img_tensor)
    self.mem_backgrounds_norm = torch.norm(self.mem_backgrounds, 2, 1)
    # load memory from apples #
    # get file list of apple images
    list = os.listdir(self.root + 'food_memory/data/256apples')
    # randomly choose #mem_size images from the list
    rand_id = torch.randperm(len(list))
    for i in xrange(mem_size):
      img_path = list[rand_id[i]]
      img = imread(self.root + 'food_memory/data/256apples' + '/' + img_path)
      img_c = img[128 - 48:128 + 48, 128 - 48:128 + 48, :]      
      img_r = imresize(img_c, [64, 64])    
      img_r = np.transpose(img_r / 255, (2, 0, 1))
      img_tensor = torch.from_numpy(img_r).float().view(1, mem_dim)
      self.mem_apples.index_copy_(0, torch.LongTensor([i]), img_tensor)
    self.mem_apples_norm = torch.norm(self.mem_apples, 2, 1)

    # load memory from lemons #
    # get file list of lemon images
    list = os.listdir(self.root + 'food_memory/data/256lemons')
    # randomly choose #mem_size images from the list
    rand_id = torch.randperm(len(list))
    for i in xrange(mem_size):
      img_path = list[rand_id[i]]
      img = imread(self.root + 'food_memory/data/256lemons' + '/' + img_path)
      img_c = img[128 - 48:128 + 48, 128 - 48:128 + 48, :]      
      img_r = imresize(img_c, [64, 64])    
      img_r = np.transpose(img_r / 255, (2, 0, 1))     
      img_tensor = torch.from_numpy(img_r).float().view(1, mem_dim)
      self.mem_lemons.index_copy_(0, torch.LongTensor([i]), img_tensor)
    self.mem_lemons_norm = torch.norm(self.mem_lemons, 2, 1)

  def lookup(self, x):
    # compute the distance between x and memories #
    # compute distance between x and memory in mem_backgrounds
    # print(self.mem_backgrounds.size())
    # print(x.view(self.mem_dim, 1).size())    
    x_v = x.view(self.mem_dim, 1)
    x_v_norm = torch.norm(x_v, 2)
    source = torch.FloatTensor(self.mem_size, 1).fill_(0)
    score_b = torch.addcdiv(source, 1, self.mem_backgrounds.mm(x_v), self.mem_backgrounds_norm) / x_v_norm
    score_a = torch.addcdiv(source, 1, self.mem_apples.mm(x_v), self.mem_apples_norm) / x_v_norm
    score_l = torch.addcdiv(source, 1, self.mem_lemons.mm(x_v), self.mem_lemons_norm) / x_v_norm

    scores = []
    scores.append(torch.mean(score_b))
    scores.append(torch.mean(score_a))
    scores.append(torch.mean(score_l))

    return scores

  def update(self, x, mem_type):
    if mem_type == 'background':
      return x
  def name(self):
    return 'foodmem'    



#define perception network for recognizing apples, lemons and walls
class FoodNet(nn.Module):
  def __init__(self):
    super(FoodNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 8, kernel_size=3)
    self.conv1.weight.data.normal_(0, 0.01)
    self.conv1.bias.data.fill_(0)

    self.conv2 = nn.Conv2d(8, 8, kernel_size=3)
    self.conv2.weight.data.normal_(0, 0.01)
    self.conv2.bias.data.fill_(0)

    self.conv3 = nn.Conv2d(8, 8, kernel_size=3)        
    self.conv3.weight.data.normal_(0, 0.01)
    self.conv3.bias.data.fill_(0)

    self.fc1 = nn.Linear(8 * 6 * 6, 8 * 2 * 2)
    self.fc1.weight.data.normal_(0, 0.01)
    self.fc1.bias.data.fill_(0)    
    #self.drop1 = nn.Dropout(0.5)

    self.fc2 = nn.Linear(8 * 2 * 2, 3)
    self.fc2.weight.data.normal_(0, 0.01)
    self.fc2.bias.data.fill_(0)   

    self.bn1 = nn.BatchNorm2d(8)
    self.bn2 = nn.BatchNorm2d(8)
    self.bn3 = nn.BatchNorm2d(8)

    self.ceriation = nn.CrossEntropyLoss()
  def forward(self, x, target):
    x = F.max_pool2d(self.conv1(x), 2)    
    x = self.bn1(x)        
    x = F.relu(x)

    x = F.max_pool2d(self.conv2(x), 2)    
    x = self.bn2(x)        
    x = F.relu(x)

    x = F.max_pool2d(self.conv3(x), 2)    
    x = self.bn3(x)        
    x = F.relu(x)

    x = x.view(-1, 8 * 6 * 6)
    x = self.fc1(x)    
    x = F.relu(x)
    x = self.fc2(x)
    loss =  self.ceriation(x, target)
    x = F.softmax(x)
    return x, loss
  def name(self):
    return 'foodnet' 


#class FoodMemory():


class Squirrel(object):
  """A random agent using spring-like forces for its action evolution."""

  def __init__(self, action_spec, width, height, use_knowledge, ai_module):

    ACTIONS = {
      'look_right': _action(20, 0, 0, 0, 0, 0, 0),
      'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
      'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
      'forward': _action(0, 0, 0, 1, 0, 0, 0),
      # 'backward': _action(0, 0, 0, -1, 0, 0, 0),      
      'look_left': _action(-20, 0, 0, 0, 0, 0, 0)
    }
    self.ACTION_LIST = ACTIONS.values()
    self.use_knowledge = use_knowledge
    self.ai_module = ai_module
    self.rewards_one_epis = 0
    self.rewards = 0
    self.depth_seq = []
    self.foodpred_seq = []
    self.frames = torch.Tensor()
    self.frames_np = []
    self.root = '/home/jwyang/Researches/lab/RLCourseProject/'

    # define the policy network, taking observation as input, 
    # and action probability as output
    # self.policynet = PolicyNet(width, height, len(self.ACTION_LIST))
    self.policynet = torch.load(self.root + 'models/policy_epoch_49.pth')
    # self.policynet.cuda()
    # self.optimizer = optim.Adam(self.policynet.parameters(), lr=2e-4)
    self.optimizer = optim.RMSprop(self.policynet.parameters(), lr=1e-2, alpha=0.95, eps=0.01)
    
    if ai_module == 'p':
      ### load foodnet          
      self.foodnet = FoodNet().float()
      if use_knowledge == True:
        checkpoint = torch.load(self.root + 'food_classifier/foodnet_13')    
        self.foodnet.load_state_dict(checkpoint)
        self.foodnet.eval()
    elif ai_module == 'm':
      ### load food memory
      self.foodmem = FoodMem(100, 64 * 64 * 3)
      if use_knowledge == True:
        self.foodmem.loadmem()
    else:
      print('no ai module is used, pure RL learning agent')


    # define plot
    self.fig = plt.figure()

  def select_action(self, state):
    probs = self.policynet(Variable(state))    
    print(probs)
    action = probs.multinomial()
    self.policynet.action_prob_seq.append(probs)
    self.policynet.action_seq.append(action)
    return action.data

  def recog_food(self, state):    
    if self.ai_module == 'p':
      labels_food = torch.LongTensor(1).fill_(0)    
      pred, _ = self.foodnet(Variable(state), Variable(labels_food))
      self.foodpred_seq.append(pred)
      #1:apple, #2:background, #3:lemon
      print(pred.data)
      return pred
    elif self.ai_module == 'm':
      pred = self.foodmem.lookup(state)    
      print(pred)
      pred = pred - pred.min()

      return pred[1]
  def step_policy(self, reward, frame, t):
    """Gets an image state and a reward, returns an action."""
    # compute reward based on the depth map
    # we do not want the agent get too close 
    # to the walls or other obstacles
    img_height = frame.shape[0]
    img_width = frame.shape[1]
    frame_rgb = frame[:,:,0:3]
    frame_depth = frame[:,:,3]
    self.rewards_one_epis += reward

    if self.use_knowledge == True:
      # exit()
      # perform augmentation on frame_rgb
      frame_c = frame_rgb[128 - 48:128 + 48, 128 - 48:128 + 48, :]
      frame_r = imresize(frame_c, [64, 64])    
      frame_t = np.transpose(frame_r / 255, (2, 0, 1))
      state4fn = torch.from_numpy(frame_t).float().unsqueeze(0)    
      p = self.recog_food(state4fn)
      if (p.data[0,0] - p.data[0,1]) > 0.2:
        reward += p.data[0,0]
      elif (p.data[0,1] - p.data[0,0]) > 0.2:
        if reward == 1:
          print('conflict! get food without seeing it')
          reward = 0
        reward -= p.data[0,1]

    frame_d = imresize(frame_rgb, 0.5)    
    frame_t = np.transpose(frame_d / 255, (2, 0, 1))
    state = torch.from_numpy(frame_t).float().unsqueeze(0)
    if t % 3 == 0:
      self.frames = torch.cat([self.frames, state], 0)
      self.frames_np.append(frame_rgb)

    self.rewards += reward
    self.policynet.reward_seq.append(reward)
    action = self.select_action(state)

    return self.ACTION_LIST[action[0, 0]]

  def clip_action(self, action):
    return np.clip(action, self.mins, self.maxs).astype(np.intc)

  def finish_episode(self):
    """update policy based on the results in one episode"""
    R = 0
    rewards = []
    labels_food = torch.LongTensor(len(self.policynet.reward_seq)).fill_(0)    
    t = len(self.policynet.reward_seq) - 1
    for r in self.policynet.reward_seq[::-1]:
      R = r + self.policynet.gamma * R
      rewards.insert(0, R)
      if r == 1:
        for tp in range(t - 5, t):
          labels_food[tp] = 1
      elif r == -1:
        for tp in range(t - 5, t):
          labels_food[tp] = 2     
      t = t - 1
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for action, r, p in zip(self.policynet.action_seq, rewards, self.policynet.action_prob_seq):
      action.reinforce(r)
    self.optimizer.zero_grad()
    autograd.backward(self.policynet.action_seq, [None for _ in self.policynet.action_seq])
    self.optimizer.step()

    # train food recognizer
    # frames_var = Variable(self.frames)
    # y_pred = self.foodrecognet(frames_var)
    # criterion = nn.CrossEntropyLoss()
    # labels_food_var = Variable(labels_food)
    # loss = criterion(y_pred, labels_food_var)
    # self.optimizer_fd.zero_grad()
    # loss.backward()
    # self.optimizer_fd.zero_grad()

    # save frames to gif
    # imageio.mimsave(self.root + 'test.gif', self.frames_np)
    self.rewards_one_epis = 0
    # plot reward curve
    del self.policynet.reward_seq[:]
    del self.policynet.action_prob_seq[:]
    del self.policynet.action_seq[:]
    del self.depth_seq[:]
    self.frames = torch.Tensor()

  def checkpoint(self, i_episode):
    save_path = self.root + "models/policy_epoch_{}.pth".format(i_episode)
    torch.save(self.policynet, save_path)
    print("Polocy saved to {}".format(save_path))

    # if self.use_knowledge == False:
    #   save_path = self.root + "models/foodnet_epoch_{}.pth".format(i_episode)
    #   torch.save(self.foodnet, save_path)
    #   print("Foodnet saved to {}".format(save_path))

def run(episode, length, width, height, fps, level, use_knowledge, ai_module):
  """Spins up an environment and runs the random agent."""
  env = deepmind_lab.Lab(
      level, ['RGBD_INTERLACED'],
      config={
          'fps': str(fps),
          'width': str(width),
          'height': str(height)
      })
  # Starts the random spring agent. As a simpler alternative, we could also
  # use DiscretizedRandomAgent().

  agent = Squirrel(env.action_spec(), width, height, use_knowledge, ai_module)

  reward = 0
  episodes = list()
  reward_episodes = list()

  for i_episode in xrange(episode):
    print("episode: ", i_episode)
    env.reset()
    for t in xrange(length):
      if not env.is_running():
        print('Environment stopped early')      
        # env.reset()
        # agent.reset()
        break
      obs = env.observations()
      #if t % 3 == 0:
      #  reward_p = reward_p + reward
      action = agent.step_policy(reward, obs['RGBD_INTERLACED'], t)      
      # imsave(self.root + 'images/' + str(t) + '.png', frame_rgb)
      # reward_p = 0
      reward = env.step(action, num_steps=1)

    print('Finished after %i steps. Total reward received is %f'
      % (length, agent.rewards_one_epis))
    episodes.append(i_episode)
    reward_episodes.append(agent.rewards_one_epis)  
    agent.finish_episode()
    print('Finished one episodes. Save policy network')  
    agent.checkpoint(i_episode)
    # exit()

  plt.plot(episodes, reward_episodes, lw=1)
  plt.savefig(agent.root + 'reward_curve_' + ai_module + '.png')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--episode', type=int, default=100,
                      help='Number of episodes to run the agent')
  parser.add_argument('--length', type=int, default=1000,
                      help='Number of steps to run the agent')
  parser.add_argument('--width', type=int, default=80,
                      help='Horizontal size of the observations')
  parser.add_argument('--height', type=int, default=80,
                      help='Vertical size of the observations')
  parser.add_argument('--fps', type=int, default=60,
                      help='Number of frames per second')
  parser.add_argument('--runfiles_path', type=str, default=None,
                      help='Set the runfiles path to find DeepMind Lab data')
  #parser.add_argument('--level_script', type=str, default='tests/new_map',
  #                    help='The environment level script to load')
  parser.add_argument('--level_script', type=str, default='tests/demo_map',
                      help='The environment level script to load')  
  parser.add_argument('--use_knowledge', type=bool, default=True,
                      help='Determine whether using prior knoledge')
  parser.add_argument('--ai_module', type=str, default='p',
                      help='Determine which AI module to use')
  parser.add_argument('--save_frames', type=bool, default=False,
                      help='Determine whether to save frames')  
  parser.add_argument('--save_rewards', type=bool, default=False,
                      help='Determine whether to save rewards')  

  args = parser.parse_args()
  if args.runfiles_path:
    deepmind_lab.set_runfiles_path(args.runfiles_path)

  run(args.episode, args.length, args.width, args.height, args.fps, args.level_script, 
    args.use_knowledge, args.ai_module)
