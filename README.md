# Pong

![License Badge](https://img.shields.io/github/license/DebarghaG/Tansen?style=plastic)
![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)

I’ve used Deep Q Learning to learn control policies to play the game of pong, directly from
visual data. The model is based on a Convolutional Neural Network that learns to use the input raw pixel
data , to estimate a value function that allows us to approximate the future rewards, for any given action.
The exact same model has been trained in different environments, and plays a near perfect game, against a
perfect rival bot, as well as against a human. Experiments have also been conducted to better understand
and evaluate how our trained agent responds to dynamic changes in the environment.

Acknowledgements :
1. Machine Learning with Phil YTC Channel -> Great resources
2. The Coding Train for PyGame Tutorials (For the simulator)
3. Countless Video Tutorials on Youtube, on whom this work is based
