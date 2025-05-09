import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import random
import gymnasium as gym
import numpy as np
import tensorflow as tf # Still useful for tf.reshape or other tf utilities if needed
from collections import deque
import keras # Using Keras 3.x
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop # Keras 3 optimizers

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
    model.compile(loss='mse', optimizer=RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

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
                
                # Extract cart position (x) and velocity from the state
                cart_position = next_state[0][0]
                cart_velocity = next_state[0][1]
                
                # Initialize variables for tracking on first step
                if i == 0:
                    self.max_position_deviation = 0
                    self.previous_position = 0
                
                # Track max position deviation for this episode
                self.max_position_deviation = max(self.max_position_deviation, abs(cart_position))
                
                # Calculate how much closer to center we've moved (positive if moving toward center)
                movement_toward_center = abs(self.previous_position) - abs(cart_position)
                
                # 1. Position components
                # 1a. "Comfort zone" bonus for staying near center (positive reinforcement)
                if abs(cart_position) < 0.5:
                    center_bonus = 0.5 * (0.5 - abs(cart_position))  # Max +0.25 at center, tapering to 0 at ±0.5
                else:
                    center_bonus = 0
                
                # 1b. Progressive position penalties (increasingly harsh as you get further out)
                if abs(cart_position) < 0.8:
                    # Mild penalty close to center
                    position_penalty = -0.5 * (cart_position ** 2)
                elif abs(cart_position) < 1.5:
                    # Medium penalty in middle region
                    position_penalty = -1.0 * (cart_position ** 2)
                else:
                    # Severe penalty in outer regions
                    position_penalty = -2.0 * (cart_position ** 2)
                
                # 2. Velocity components
                # Directional velocity penalty: positive velocity moves right, negative moves left
                velocity_direction = 1 if cart_position > 0 else -1
                moving_away_from_center = (velocity_direction * cart_velocity > 0)
                
                # 2a. Velocity penalty
                if moving_away_from_center:
                    # Stronger penalty when moving away from center
                    velocity_factor = 4.0 if abs(cart_position) > 1.0 else 3.0  # Even stronger when already far out
                    velocity_penalty = -velocity_factor * (cart_velocity ** 2)
                    
                    # Ultra-severe penalty for high velocity away from center in danger zone
                    if abs(cart_position) > 1.2 and abs(cart_velocity) > 1.0:
                        velocity_penalty *= 1.5
                        print(f"DANGER: High velocity {cart_velocity:.2f} away from center at position {cart_position:.2f}")
                else:
                    # Small penalty for velocity magnitude when moving toward center
                    velocity_penalty = -0.3 * (cart_velocity ** 2)
                
                # 3. Recovery components
                recovery_bonus = 0
                
                # 3a. Any movement toward center is good (lower threshold to 0.5)
                if abs(self.previous_position) > 0.5 and movement_toward_center > 0:
                    # Base recovery bonus scaled by position and improvement
                    recovery_factor = 1.0
                    
                    # Progressively stronger recovery bonuses based on position
                    if abs(self.previous_position) > 1.5:
                        recovery_factor = 5.0  # Major recovery
                    elif abs(self.previous_position) > 1.0:
                        recovery_factor = 3.0  # Medium recovery
                    
                    recovery_bonus = recovery_factor * abs(self.previous_position) * movement_toward_center
                    
                    # Print logs for significant recoveries
                    if abs(self.previous_position) > 1.5 and movement_toward_center > 0.05:
                        print(f"MAJOR RECOVERY: +{recovery_bonus:.2f} for moving from {self.previous_position:.2f} to {cart_position:.2f}")
                    elif abs(self.previous_position) > 1.0 and movement_toward_center > 0.05:
                        print(f"Recovery bonus: +{recovery_bonus:.2f} for moving from {self.previous_position:.2f} to {cart_position:.2f}")
                
                # Remember position for next step
                self.previous_position = cart_position
                
                # For debugging velocities
                if abs(cart_velocity) > 1.5 and i % 10 == 0:  # Only log every 10th step to reduce spam
                    if moving_away_from_center:
                        print(f"HIGH VELOCITY AWAY FROM CENTER: {cart_velocity:.2f} at position {cart_position:.2f}")
                    else:
                        print(f"High velocity toward center: {cart_velocity:.2f} at position {cart_position:.2f}")
                
                if not terminated or i == self.env._max_episode_steps-1:
                    # Combine all reward components
                    final_reward = reward + center_bonus + position_penalty + velocity_penalty + recovery_bonus
                    
                    # Add extreme edge penalty as a last resort safety net
                    if abs(cart_position) > 2.0:
                        edge_penalty = -10.0 * (abs(cart_position) - 2.0) 
                        final_reward += edge_penalty
                        print(f"EDGE PENALTY: {edge_penalty:.2f} at position {cart_position:.2f}")
                    
                    # Log total reward components occasionally for debugging
                    if i % 100 == 0:
                        print(f"Position: {cart_position:.2f}, Rewards: [base: {reward:.2f}, center: {center_bonus:.2f}, " +
                              f"pos_penalty: {position_penalty:.2f}, vel_penalty: {velocity_penalty:.2f}, " +
                              f"recovery: {recovery_bonus:.2f}]")
                    
                    reward = final_reward
                else:
                    # Failure penalty
                    reward = -100
                self.remember(state, action, reward, next_state, terminated)
                #print("Episode: {}, Step: {}, Action: {}, Reward: {}, Info: {}, State: {}, Terminated: {}".format(e, i, action, reward, info, state, terminated))
                state = next_state
                i += 1
                # Force termination if episode reaches or exceeds 500 steps
                if i >= 500:
                    print("Episode reached 500 steps - Environment solved!")
                    print("Saving trained model as cartpole-dqn-solved.keras")
                    self.save("data/cart_pole/cartpole-dqn-solved.keras")
                    terminated = True
                    # Break out of the episode loop immediately
                    break
                
                if terminated:
                    # Single consolidated log line with all episode information
                    print("episode: {}/{}, score: {}, max_pos_dev: {:.2f}, final_pos: {:.2f}, e: {:.2f}".format(
                        e, self.EPISODES, i, self.max_position_deviation, cart_position, self.epsilon))
                    
                    # Reset max position tracking for next episode
                    self.max_position_deviation = 0
                self.replay()

    def test(self):
        self.load("data/cart_pole/cartpole-dqn.keras")
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
    # Create data directory if it doesn't exist
    os.makedirs("data/cart_pole", exist_ok=True)
    
    agent = DQNAgent()
    agent.run()
    
    # Save final model after all episodes complete
    print("Training complete. Saving final model to data/cart_pole/cartpole-dqn-final.keras")
    agent.save("data/cart_pole/cartpole-dqn-final.keras")
    
    #agent.test()
