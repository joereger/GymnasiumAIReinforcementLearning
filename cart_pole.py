import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import random
import gymnasium as gym
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.optimizers.legacy import RMSprop

def LeModel(input_shape, action_space):
    X_input = Input(input_shape)

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)

    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    
    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    # Output Layer with # of actions: 2 nodes (left, right)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs = X_input, outputs = X, name='CartPole_DQN_Model')
    model.compile(loss='mse', optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()
    return model

class DQNAgent:
    def __init__(self):
        self.env = gym.make('CartPole-v1', render_mode="human")
        #self.env = gym.make('CartPole-v1') # No visualization, so sad but so fast
        # by default, CartPole-v1 has max episode steps = 500
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = 1000
        self.memory = deque(maxlen=2000)
        
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 1000

        # create main model
        self.model = LeModel(input_shape=(self.state_size,), action_space = self.action_size)

    def remember(self, state, action, reward, next_state, terminated):
        self.memory.append((state, action, reward, next_state, terminated))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state, verbose = 0))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, terminated = [], [], []

        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            terminated.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state, verbose = 0)
        target_next = self.model.predict(next_state, verbose = 0)

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if terminated[i]:
                target[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)
            
    def run(self):
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state[0], [1, self.state_size])
            terminated = False
            i = 0
            while not terminated:
                self.env.render()
                action = self.act(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not terminated or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, terminated)
                #print("Episode: {}, Step: {}, Action: {}, Reward: {}, Info: {}, State: {}, Terminated: {}".format(e, i, action, reward, info, state, terminated))
                state = next_state
                i += 1
                if terminated:                   
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, self.EPISODES, i, self.epsilon))
                    if i >= 500:
                        print("Saving trained model as cartpole-dqn.keras")
                        self.save("data/cartpole-dqn.keras")
                        return
                self.replay()

    def test(self):
        self.load("data/cartpole-dqn.keras")
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state[0], [1, self.state_size])
            terminated = False
            i = 0
            while not terminated:
                self.env.render()
                action = np.argmax(self.model.predict(state, verbose = 0))
                next_state, reward, terminated, truncated, info = self.env.step(action)
                state = np.reshape(next_state[0], [1, self.state_size])
                i += 1
                if terminated:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    break

if __name__ == "__main__":
    agent = DQNAgent()
    agent.run()
    #agent.test()