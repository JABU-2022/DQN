import pygame
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.models import load_model
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from MaterialEnv.material_env import MaterialEnv

# Initialize Pygame
pygame.init()

# Define constants
GRID_SIZE = 5
CELL_SIZE = 100
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
COLORS = {
    'Cotton': (255, 255, 255),
    'Polyester': (0, 255, 0),
    'Wool': (0, 0, 255),
    'Nylon': (255, 0, 0)
}

# Create the display window
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Material Environment Visualization")

def draw_grid(env):
    screen.fill(WHITE)
    # Draw the grid
    for x in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (x, 0), (x, WINDOW_SIZE))
    for y in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (0, y), (WINDOW_SIZE, y))
    
    # Draw materials
    for material, pos in env.materials_pos.items():
        pygame.draw.rect(screen, COLORS[material], (pos[1] * CELL_SIZE, pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    # Draw agent
    agent_pos = env.agent_pos
    pygame.draw.rect(screen, BLACK, (agent_pos[1] * CELL_SIZE, agent_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

def visualize(env, agent, episodes=1):
    clock = pygame.time.Clock()
    running = True

    for episode in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            
            if not running:
                break
            
            # Get action from the agent
            action = agent.forward(obs)
            obs, reward, done, _ = env.step(action)
            draw_grid(env)
            pygame.display.flip()
            clock.tick(1)  # Slow down the rendering to see the movement

    pygame.quit()

if __name__ == "__main__":
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

    # Run the visualization
    visualize(env, dqn)
