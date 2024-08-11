import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from material_env import MaterialEnv

# Create an instance of the environment
env = MaterialEnv()
states = env.observation_space.shape
actions = env.action_space.n

# Build the model
model = Sequential()
model.add(Flatten(input_shape=(1,) + states))
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(actions, activation='linear'))

# Configure the agent with the saved weights
policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
dqn.compile(tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae'])

# Load the trained weights
dqn.load_weights('dqn_material_env_weights.h5f')

# Test the agent and render the environment
for episode in range(5):
    obs = env.reset()
    done = False
    while not done:
        action = dqn.forward(obs)
        obs, reward, done, _ = env.step(action)
        env.render()
        print(f"Episode: {episode + 1}, Action: {action}, Reward: {reward}")