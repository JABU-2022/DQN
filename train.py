import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from MaterialEnv.material_env import MaterialEnv

# Create environment
env = MaterialEnv()

# Define the model architecture
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))

# Configure the agent with higher exploration
policy = EpsGreedyQPolicy(eps=0.3)  # Start with higher exploration rate
memory = SequentialMemory(limit=3000, window_length=1)
dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=env.action_space.n, nb_steps_warmup=10, target_model_update=1e-2)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

# Train the agent
dqn.fit(env, nb_steps=3000, visualize=False, verbose=2)

# Save the model weights
dqn.save_weights('dqn_material_env_weights.h5f', overwrite=True)