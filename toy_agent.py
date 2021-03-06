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



class DiscretizedRandomAgent(object):
  """Simple agent for DeepMind Lab."""

  ACTIONS = {
      'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
      'look_right': _action(20, 0, 0, 0, 0, 0, 0),
      'look_up': _action(0, 10, 0, 0, 0, 0, 0),
      'look_down': _action(0, -10, 0, 0, 0, 0, 0),
      'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
      'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
      'forward': _action(0, 0, 0, 1, 0, 0, 0),
      'backward': _action(0, 0, 0, -1, 0, 0, 0),
      'fire': _action(0, 0, 0, 0, 1, 0, 0),
      'jump': _action(0, 0, 0, 0, 0, 1, 0),
      'crouch': _action(0, 0, 0, 0, 0, 0, 1)
  }

  ACTION_LIST = ACTIONS.values()

  def step(self, unused_reward, unused_image):
    """Gets an image state and a reward, returns an action."""
    return random.choice(DiscretizedRandomAgent.ACTION_LIST)

class Net(nn.Module):
    def __init__(self, width, height, actions):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)        
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, actions)
        self.gamma = 0.99

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 4))
        x = F.relu(F.max_pool2d(self.conv3(x), 4))
        x = x.view(-1, 64 * 2 * 2)
        x = F.relu(self.fc1(x))        
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # return F.softmax(x)
        return F.sigmoid(x)

class SpringAgent(object):
  """A random agent using spring-like forces for its action evolution."""

  def __init__(self, action_spec, width, height):
    self.action_spec = action_spec
    print('Starting random spring agent. Action spec:', action_spec)

    self.omega = np.array([
        0.1,  # look left-right
        0.1,  # look up-down
        0.1,  # strafe left-right
        0.1,  # forward-backward
        0.0,  # fire
        0.0,  # jumping
        0.0  # crouching
    ])

    self.velocity_scaling = np.array([2.5, 2.5, 0.00, 0.01, 1, 1, 1])
    # self.velocity_scaling = np.array([2.5, 0, 0.01, 0.01, 1, 1, 1])

    self.indices = {a['name']: i for i, a in enumerate(self.action_spec)}
    self.mins = np.array([a['min'] for a in self.action_spec])
    self.maxs = np.array([a['max'] for a in self.action_spec])
    self.reset()

    self.rewards = 0

    self.reward_seq = []
    self.action_prob_seq = []
    self.action_seq = []

    # define the policy network, taking observation as input, 
    # and action probability as output
    torch.cuda.set_device(0)
    self.policynet = Net(width, height, len(self.action_spec))
    # self.policynet.cuda()
    self.optimizer = optim.Adam(self.policynet.parameters(), lr=1e-2)

  def critically_damped_derivative(self, t, omega, displacement, velocity):
    """Critical damping for movement.

    I.e., x(t) = (A + Bt) \exp(-\omega t) with A = x(0), B = x'(0) + \omega x(0)

    See
      https://en.wikipedia.org/wiki/Damping#Critical_damping_.28.CE.B6_.3D_1.29
    for details.

    Args:
      t: A float representing time.
      omega: The undamped natural frequency.
      displacement: The initial displacement at, x(0) in the above equation.
      velocity: The initial velocity, x'(0) in the above equation

    Returns:
       The velocity x'(t).
    """
    a = displacement
    b = velocity + omega * displacement
    return (b - omega * t * (a + t * b)) * np.exp(-omega * t)

  def select_action(self, frame):
    frame_d = imresize(frame, 0.5)
    frame_t = np.transpose(frame_d, (2, 0, 1))
    state = torch.from_numpy(frame_t).float().unsqueeze(0)
    probs = self.policynet(Variable(state))
    self.action_prob_seq.append(probs)
    # action = probs.multinomial()
    # print(action.data)
    # model.saved_actions.append(action)
    return probs.squeeze(0)

  def step(self, reward, unused_frame):
    """Gets an image state and a reward, returns an action."""
    self.rewards += reward

    action = (self.maxs - self.mins) * np.random.random_sample(
        size=[len(self.action_spec)]) + self.mins

    # print(action)
    # Compute the 'velocity' 1 time unit after a critical damped force
    # dragged us towards the random `action`, given our current velocity.
    self.velocity = self.critically_damped_derivative(1, self.omega, action,
                                                      self.velocity)

    # Since walk and strafe are binary, we need some additional memory to
    # smoothen the movement. Adding half of action from the last step works.
    self.action = self.velocity / self.velocity_scaling + 0.5 * self.action

    # Fire with p = 0.01 at each step
    self.action[self.indices['FIRE']] = int(np.random.random() > 0.99)

    # Jump/crouch with p = 0.005 at each step
    self.action[self.indices['JUMP']] = int(np.random.random() > 0.995)
    self.action[self.indices['CROUCH']] = int(np.random.random() > 0.995)

    # Clip to the valid range and convert to the right dtype
    return self.clip_action(self.action)

  def step_policy(self, reward, unused_frame):
    """Gets an image state and a reward, returns an action."""
    self.rewards += reward
    self.reward_seq.append(reward)

    action_probs = self.select_action(unused_frame)
    action_probs = action_probs.data.numpy()

    action = (self.maxs - self.mins) * action_probs + self.mins

    # Compute the 'velocity' 1 time unit after a critical damped force
    # dragged us towards the random `action`, given our current velocity.
    self.velocity = self.critically_damped_derivative(1, self.omega, action,
      self.velocity)

    # Since walk and strafe are binary, we need some additional memory to
    # smoothen the movement. Adding half of action from the last step works.
    self.action = self.velocity / self.velocity_scaling + 0.5 * self.action

    # No look up and down
    self.action[1] = 0
    self.action[2] = 0

    # No backward
    if self.action[3] < 0:
      self.action[3] = 0

    # No fire with p = 0.01 at each step
    self.action[self.indices['FIRE']] = 0

    # No jump/crouch with p = 0.005 at each step
    self.action[self.indices['JUMP']] = 0
    self.action[self.indices['CROUCH']] = 0

    # Clip to the valid range and convert to the right dtype
    self.action_seq.append(self.clip_action(self.action))
    return self.clip_action(self.action)

  def clip_action(self, action):
    return np.clip(action, self.mins, self.maxs).astype(np.intc)

  def reset(self):
    self.velocity = np.zeros([len(self.action_spec)])
    self.action = np.zeros([len(self.action_spec)])

  def finish_episode(self):
    """update policy based on the results in one episode"""
    R = 0
    rewards = []
    for r in self.reward_seq[::-1]:
      R = r + self.policynet.gamma * R
      rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    gradients = [torch.zeros(1, len(self.action_spec))] * len(self.action_seq)
    for t in xrange(len(self.reward_seq)):
        for a in np.array([0, 3]):
          # if self.action_seq[t][0][a] > 0.5:
          #   gradients[t][0][a] = -rewards[t]
          # elif self.action_seq[t][0][a] < 0.5:
          #   gradients[t][0][a] = rewards[t]          
          if self.action_seq[t][a] > 0:
            gradients[t][0][a] = -rewards[t]
          elif self.action_seq[t][a] < 0:
            gradients[t][0][a] = rewards[t]

    self.optimizer.zero_grad()
    autograd.backward(self.action_prob_seq, gradients)
    self.optimizer.step()
    del self.reward_seq[:]
    del self.action_prob_seq[:]
    del self.action_seq[:]

  def checkpoint(self, i_episode):
    save_path = "policy_epoch_{}.pth".format(i_episode)
    torch.save(self.policynet, save_path)
    print("Polocy saved to {}".format(save_path))

def run(episode, length, width, height, fps, level):
  """Spins up an environment and runs the random agent."""
  env = deepmind_lab.Lab(
      level, ['RGB_INTERLACED'],
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
    for _ in xrange(length):
      if not env.is_running():
        print('Environment stopped early')      
        # env.reset()
        # agent.reset()
        break
      obs = env.observations()
      action = agent.step_policy(reward, obs['RGB_INTERLACED'])      
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
  parser.add_argument('--level_script', type=str, default='squirrel_map',
                      help='The environment level script to load')

  args = parser.parse_args()
  if args.runfiles_path:
    deepmind_lab.set_runfiles_path(args.runfiles_path)
  run(args.episode, args.length, args.width, args.height, args.fps, args.level_script)
