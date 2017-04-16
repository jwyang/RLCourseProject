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
from scipy.misc import imsave
from scipy.misc import imresize

import deepmind_lab

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
    self.conv1 = nn.Conv2d(4, 16, kernel_size=5)
    self.conv1.weight.data.normal_(0, 0.01)
    self.conv1.bias.data.fill_(0)

    self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
    self.conv2.weight.data.normal_(0, 0.01)
    self.conv2.bias.data.fill_(0)

    self.conv3 = nn.Conv2d(32, 64, kernel_size=5)        
    self.conv3.weight.data.normal_(0, 0.01)
    self.conv3.bias.data.fill_(0)

    self.fc1 = nn.Linear(64 * 2 * 2, actions)
    self.fc1.weight.data.normal_(0, 0.01)
    self.fc1.bias.data.fill_(0)

    # self.fc2 = nn.Linear(64, actions)
    self.gamma = 0.99
    self.reward_seq = []
    self.action_prob_seq = []
    self.action_seq = []

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2(x), 4))
    x = F.relu(F.max_pool2d(self.conv3(x), 4))
    x = x.view(-1, 64 * 2 * 2)
    # x = F.relu(self.fc1(x))        
    # x = F.dropout(x, training=self.training)
    x = self.fc1(x)
    return F.softmax(x)

# # define memory class for storing short-term memory
# class Mmeory():
#   def __init__(self, mem_size, mem_dim):

#define perception class for recognizing apples and predators
class RecognizeNet(nn.Module):
  def __init__(self):
    super(RecognizeNet, self).__init__()
    self.conv1 = nn.Conv2d(4, 16, kernel_size=5)
    self.conv1.weight.data.normal_(0, 0.01)
    self.conv1.bias.data.fill_(0)

    self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
    self.conv2.weight.data.normal_(0, 0.01)
    self.conv2.bias.data.fill_(0)

    self.conv3 = nn.Conv2d(32, 64, kernel_size=5)        
    self.conv3.weight.data.normal_(0, 0.01)
    self.conv3.bias.data.fill_(0)

    self.fc1 = nn.Linear(64 * 2 * 2, 2)
    self.fc1.weight.data.normal_(0, 0.01)
    self.fc1.bias.data.fill_(0)    

    self.pred_seq = []
    self.reward_seq = []

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2(x), 4))
    x = F.relu(F.max_pool2d(self.conv3(x), 4))
    x = x.view(-1, 64 * 2 * 2)
    x = self.fc1(x)
    return F.softmax(x)

class SpringAgent(object):
  """A random agent using spring-like forces for its action evolution."""

  def __init__(self, action_spec, width, height):

    ACTIONS = {
      'look_right': _action(20, 0, 0, 0, 0, 0, 0),
      'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
      'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
      'forward': _action(0, 0, 0, 1, 0, 0, 0),
      'backward': _action(0, 0, 0, -1, 0, 0, 0),      
      'look_left': _action(-20, 0, 0, 0, 0, 0, 0)
    }
    self.ACTION_LIST = ACTIONS.values()
    self.rewards = 0
    self.depth_seq = []
    self.frames = torch.Tensor()
    self.root = '/home/jwyang/Researches/lab/RLCourseProject/'
    # define the policy network, taking observation as input, 
    # and action probability as output
    # torch.cuda.set_device(0)
    self.policynet = PolicyNet(width, height, len(self.ACTION_LIST))
    self.foodrecognet = RecognizeNet()
    self.predrecognet = RecognizeNet()
    # self.policynet = torch.load('policy_epoch_10.pth')
    # self.policynet.cuda()
    self.optimizer = optim.Adam(self.policynet.parameters(), lr=5e-4)
    self.optimizer_fd = optim.Adam(self.foodrecognet.parameters(), lr=1e-2)
    self.optimizer_pd = optim.Adam(self.predrecognet.parameters(), lr=1e-2)


  def select_action(self, state):
    probs = self.policynet(Variable(state))    
    print(probs)
    action = probs.multinomial()
    self.policynet.action_prob_seq.append(probs)
    self.policynet.action_seq.append(action)
    return action.data

  def recog_food(self, state):
    pred = self.foodrecognet(Variable(state))
    self.foodrecognet.pred_seq.append(pred)
    isfood = pred.data[0, 1]     
    return isfood

  def step_policy(self, reward, frame, t):
    """Gets an image state and a reward, returns an action."""
    # compute reward based on the depth map
    # we do not want the agent get too close 
    # to the walls or other obstacles
    # imsave(self.root + 'images/output_' + str(t) + '.jpg', frame[:,:,0:3])
    frame_depth = frame[:,:,3]
    self.depth_seq.append(frame_depth.mean())
    if frame_depth.mean() / 255 < 0.1:
      reward = reward + (frame_depth.mean() / 255 - 1)

    frame = 2 * (frame / 255 - 0.5)
    frame_d = imresize(frame, 0.5)    
    frame_t = np.transpose(frame_d, (2, 0, 1))
    state = torch.from_numpy(frame_t).float().unsqueeze(0)
    self.frames = torch.cat([self.frames, state], 0)

    # judge whether the squirrel see the food
    # isfood = self.recog_food(state)
    # self.foodrecognet.reward_seq.append(max(0, isfood - 0.6))    

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
      # r_food = self.foodrecognet.reward_seq[t]
      R = r + self.policynet.gamma * R
      rewards.insert(0, R)
      if r == 1:
        for tp in range(t - 5, t):
          labels_food[tp] = 1      
      t = t - 1
    print(rewards)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for action, r in zip(self.policynet.action_seq, rewards):
      action.reinforce(r)
    self.optimizer.zero_grad()
    autograd.backward(self.policynet.action_seq, [None for _ in self.policynet.action_seq])
    self.optimizer.step()

    # train food recognizer
    #frames_var = Variable(self.frames)
    #y_pred = self.foodrecognet(frames_var)
    #criterion = nn.CrossEntropyLoss()
    #labels_food_var = Variable(labels_food)
    #loss = criterion(y_pred, labels_food_var)
    #self.optimizer_fd.zero_grad()
    #loss.backward()
    #self.optimizer_fd.zero_grad()

    del self.policynet.reward_seq[:]
    del self.policynet.action_prob_seq[:]
    del self.policynet.action_seq[:]
    del self.depth_seq[:]
    self.frames = torch.Tensor()

    del self.foodrecognet.pred_seq[:]
    del self.foodrecognet.reward_seq[:]

  def checkpoint(self, i_episode):
    save_path = self.root + "models/policy_epoch_{}.pth".format(i_episode)
    torch.save(self.policynet, save_path)
    print("Polocy saved to {}".format(save_path))

    save_path = self.root + "models/foodnet_epoch_{}.pth".format(i_episode)
    torch.save(self.foodrecognet, save_path)
    print("Foodnet saved to {}".format(save_path))

def run(episode, length, width, height, fps, level):
  """Spins up an environment and runs the random agent."""
  env = deepmind_lab.Lab(
      level, ['RGBD_INTERLACED'],
      config={
          'fps': str(fps),
          'width': str(width),
          'height': str(height)
      })

  # env.reset()

  # Starts the random spring agent. As a simpler alternative, we could also
  # use DiscretizedRandomAgent().
  agent = SpringAgent(env.action_spec(), width, height)

  reward = 0

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
      action = agent.step_policy(reward, obs['RGBD_INTERLACED'], t)      
      reward = env.step(action, num_steps=1)

    print('Finished after %i steps. Total reward received is %f'
      % (length, agent.rewards))

    agent.finish_episode()
    print('Finished one episodes. Save policy network')  
    agent.checkpoint(i_episode)

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

  args = parser.parse_args()
  if args.runfiles_path:
    deepmind_lab.set_runfiles_path(args.runfiles_path)
  run(args.episode, args.length, args.width, args.height, args.fps, args.level_script)
