# CartPole-v0

Double Q-learning with experience replay on CartPole-v0 using Keras. See [algorithm on CartPole-v0](https://gym.openai.com/evaluations/eval_4BlI2j3fQzWHEJUdDAv3w).

## Neural Network

~~~
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 128)               640
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 258
=================================================================
Total params: 898.0
Trainable params: 898
Non-trainable params: 0.0
~~~

## References

* Jaromír Janisch, [Let’s make a DQN: Implementation](https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/)
* Jaromír Janisch, [Let’s make a DQN: Debugging](https://jaromiru.com/2016/10/12/lets-make-a-dqn-debugging/)
* Jaromír Janisch, [Let’s make a DQN: Full DQN](https://jaromiru.com/2016/10/21/lets-make-a-dqn-full-dqn/)
* Keon Kim, [Deep Q Learning with Keras and Gym](https://keon.io/deep-q-learning/)
* H Van Hasselt, A Guez, D Silver, [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)
* HV Hasselt, [Double Q-learning](http://papers.nips.cc/paper/3964-double-q-learning.pdf)
* V Mnih, K Kavukcuoglu, D Silver, A Graves, [Playing atari with deep reinforcement learning](https://arxiv.org/pdf/1312.5602.pdf)