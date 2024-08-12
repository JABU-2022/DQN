import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.models import load_model
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

# Configure the agent with the policy
policy = EpsGreedyQPolicy(eps=0.0)  # No exploration, purely exploiting the learned policy
memory = SequentialMemory(limit=50000, window_length=1)

# Recreate the DQN agent with proper configurations
dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=env.action_space.n, nb_steps_warmup=10, target_model_update=1e-2)

# Compile the agent
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

# Load the weights directly into the model
model.load_weights('dqn_material_env_weights.h5f')

# Set the model weights to the DQN agent
dqn.model.set_weights(model.get_weights())

# Run the agent in the environment
def play(env, agent, episodes=1):
    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.forward(obs)  # Get action from the agent
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()  # Optional: render the environment

        print(f"Episode {episode + 1} - Total Reward: {total_reward}")

# Play the game
play(env, dqn)
