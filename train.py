import numpy as np
import gym
from material_env import MaterialEnv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# Create environment
env = MaterialEnv()
states = env.observation_space.shape
actions = env.action_space.n

# Build the model
model = Sequential()
model.add(Input(shape=states))  # Adjust shape as needed
model.add(Flatten())
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(actions, activation='linear'))

# Configure and compile the model
model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')

# Configure and compile the agent
policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
dqn.compile(optimizer=Adam(learning_rate=1e-3), metrics=['mae'])

# Train the agent
dqn.fit(env, nb_steps=5000, visualize=False, verbose=1)

# Save the model weights
dqn.save_weights('dqn_material_env_weights.h5f', overwrite=True)
