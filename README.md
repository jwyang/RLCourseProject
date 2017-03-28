# CS6795 Course Team Project

### Overview

In this project, we are simulating a situation where an animal, like squirrel, try to search for food and run away from the predators and humans. We simulate this since it is fairly normal in our campus and surroudings. To simulate such an interaction between agents and environment. We resort to use Reinforcement Learning (RL) which is super suitable for this task. For convinience, we build our model based on the open source library, [Deepmind Lab](https://github.com/deepmind/lab).

### Dependencies

1. [Deepmind Lab](https://github.com/deepmind/lab). Install deepmind lab according to the instructions in the original git repository.

2. [PyTorch](). Since we need to analyze the visual signals perceived by the agents, we use convolutional neural network (CNN) in PyTorch. Please also refer to the orignal repository for the installation.

### Train the agent

Once you have installed the above two main dependencies, you should be able to run the code to train the agents, simply using:

```bash
$ bazel run :random_agent --define headless=false -- --length=1000 --episode=20 --height=256 --width=256
```


