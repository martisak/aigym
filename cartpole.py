import gym
from gym import wrappers

import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
import copy
import matplotlib.pyplot as plt

import coloredlogs
import logging

# Things to try
# * [X] DDQN
# * [ ] Loss function
# * [X] Optimizer
# * [ ] Prioritized Experience Replay Buffer
# * [X] Loss clipping


class Agent:

    def __init__(self, env):
        self.logger = logging.getLogger("agent")
        self.logger.info("Initializing agent")
        self.env = env

        self.gamma = 0.99  # decay rate
        self.epsilon = 1  # exploration
        self.epsilon_decay = .9995
        self.epsilon_min = 0.1
        self.learning_rate = 0.001
        self.model = self.make_model(self.env)
        self.target_model = self.make_model(self.env)

        self.memory = []
        self.memory_size = 10000

        self.normalization_vector = self.env.observation_space.high
        self.normalization_vector[1] = 1
        self.normalization_vector[3] = 1

        # For evaluation
        self.interesting_states = []
        self.interesting_states.append(self.prep_input(self.env.reset(), 4))
        self.interesting_states.append(self.env.observation_space.high * 0.9)

    def load_weights(self):
        self.logger.debug("Loading weights")
        self.model.load_weights("weights.hdf5")
        self.copy_weights()

    def normalize(self, a):
        return (a)  # / np.std(a, 0)

    def make_model(self, env):

        model = Sequential()

        # No hidden layers
        model.add(
            Dense(
                128,
                input_shape=env.observation_space.shape,
                activation='relu'
            )
        )

        # model.add(Dense(128, activation='relu'))
        # model.add(Dense(16, activation='relu'))

        model.add(Dense(env.action_space.n, activation='linear'))

        opt = RMSprop(lr=self.learning_rate, clipvalue=0.1)
        model.compile(optimizer=opt, loss='mse')

        return model

    def prep_input(self, observation, n_dimension):
        return np.reshape(observation, [1, n_dimension]) / \
            self.normalization_vector

    def take_action(self, observation):

        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()

        return self.get_action(observation)

    def get_action(self, observation):
        p = self.model.predict(observation, batch_size=1)
        return np.argmax(p)

    def interesting_states_action(self):
        for s in self.interesting_states:
            p = self.model.predict(self.prep_input(s, 4), batch_size=1)
            yield p[0]

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)

        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        batch_size = min(batch_size, len(self.memory))

        batches = np.random.choice(len(self.memory), batch_size)

        X = np.zeros((len(batches), 4))
        y = np.zeros((len(batches), 2))

        for i in range(len(batches)):

            state, action, reward, next_state, done = self.memory[batches[i]]

            target = self.target_model.predict(state, batch_size=1)[0]

            target[action] = reward

            if not done:
                target[action] = reward + self.gamma * \
                    np.amax(self.target_model.predict(next_state))

            X[i] = state
            y[i] = target

        self.model.fit(self.normalize(X), y, batch_size=len(
            batches), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return self.epsilon

    def copy_weights(self):
        self.logger.debug("Copied to target network")
        self.target_model.set_weights(self.model.get_weights())


def run_episode(env, agent, replay=32, max_timesteps=1000):

    observation = agent.prep_input(env.reset(), 4)

    for t in range(max_timesteps + 1):
        # env.render()

        action = agent.take_action(observation)
        next_observation, reward, done, _ = env.step(action)
        next_observation = agent.prep_input(next_observation, 4)

        reward = -10 if done else reward

        agent.remember(
            observation, action, reward, next_observation, done)

        observation = copy.deepcopy(next_observation)

        eps = agent.replay(replay)

        if done:
            break

    return t, eps


def runningmean(x, N):
    # return np.convolve(x, np.ones((N,)) / N)[(N - 1):]
    return np.convolve(x, np.ones((N,)) / N)[:-(N - 1)]


def plotme(episodes, results, epsilon, initial_state, end_state):

    f, ax = plt.subplots(3, sharex=True, figsize=(20, 10))

    ax[0].axhline(y=195, color='r', linestyle=':', label='Solved threshold')
    ax[0].plot(list(range(episodes)), results, label='Reward')
    ax[0].plot(list(range(episodes)), runningmean(
        results, 100), label='Running average of reward')
    ax[0].set_ylabel("Score")
    ax[0].legend()

    ax[1].axhline(y=1, color='r', linestyle=':',
                  label=r'$\varepsilon_\mathrm{max}$')
    ax[1].axhline(y=agent.epsilon_min, color='g',
                  linestyle=':', label=r'$\varepsilon_\mathrm{min}$')
    ax[1].plot(list(range(episodes)), epsilon, label=r'$\varepsilon$')
    ax[1].set_ylabel(r'$\varepsilon$')
    ax[1].legend()

    ax[2].axhline(y=100, color='r', linestyle=':',
                  label='Maximum future discounted reward')
    ax[2].axhline(y=-10, color='g', linestyle=':', label='Reward for failing')

    labels = ['Initial state [left]', 'Initial state [right]']

    for y_arr, label in zip(initial_state.T, labels):
        ax[2].plot(list(range(episodes)), y_arr, label=label)

    labels = ['State close to end [left]', 'State close to end [right]']

    for y_arr, label in zip(end_state.T, labels):
        ax[2].plot(list(range(episodes)), y_arr, label=label)

    ax[2].set_ylabel("Q(s, a) for interesting states")
    ax[2].legend()

    plt.xlabel("Episode")
    plt.savefig("results.png")


if __name__ == "__main__":

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    logger = logging.getLogger(__name__)
    coloredlogs.install(level='DEBUG', logger=logger)

    logger.info("Making CartPole-v0 environment")

    env = gym.make('CartPole-v0')

    env.seed(123)
    np.random.seed(123)

    env = wrappers.Monitor(env, 'cartpole-experiment-1', force=True)

    agent = Agent(env)
    logger.debug(agent.model.summary())

    # Load pre-trained model
    # agent.load_weights()
    # agent.epsilon = agent.epsilon_min

    episodes = 200
    best = 0

    results = np.zeros(episodes)
    epsilon = np.zeros(episodes)
    initial_state = np.zeros((episodes, 2))
    end_state = np.zeros((episodes, 2))

    for i_episode in range(episodes):

        t, eps = run_episode(env, agent, replay=64, max_timesteps=200)

        if t > best:
            best = t

        logger.info("episode: {}/{}, score: {} ({}), size: {}"
                    .format(i_episode, episodes, t, best, len(agent.memory)))

        results[i_episode] = t
        epsilon[i_episode] = eps

        interesting_states = list(agent.interesting_states_action())
        initial_state[i_episode, :] = interesting_states[0]
        end_state[i_episode, :] = interesting_states[1]

        if (i_episode % 10) == 0:
            agent.copy_weights()

        if (i_episode % 50) == 0:
            logger.debug("Saving results to file.")
            plotme(episodes, results, epsilon,
                   initial_state, end_state)
            agent.model.save_weights("weights.hdf5")

    plotme(episodes, results, epsilon, initial_state, end_state)
