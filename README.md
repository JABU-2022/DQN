# Deep Q-Learning for Material Collection

## Objective

The goal of this project is to train a reinforcement learning agent using Deep Q-Learning (DQN) to navigate a 5x5 grid environment, collecting materials with varying sustainability scores while avoiding penalties for movement.

## Instructions for Setting Up the Project

### Prerequisites

- Python 3.8+
- TensorFlow 2.15+
- Keras-RL2

### Clone the Repository

To get started, clone the repository and navigate into the project directory:


git clone https://github.com/yourusername/material-dqn.git
cd material-dqn

### Create a Virtual Environment

Create and activate a virtual environment to manage dependencies:


python -m venv myenv
myenv\Scripts\activate  # On Windows
source myenv/bin/activate  # On macOS/Linux


### Install the Necessary Packages

Install the required packages using pip:



pip install gym numpy tensorflow keras-rl2


### Files

**material_env.py:** Defines the custom Gym environment for the material collection simulation.
**train.py:** Script to train the DQN agent.
**play.py:** Script to simulate the trained agent


### How to Run

### Train the Agent

To train the DQN agent, execute the following command:


python train.py

### Simulate the Trained Agent

If a simulation script is available, you can test the trained agent with:



python play.py

###Results

The trained agent effectively navigates the 5x5 grid environment, successfully collecting materials with high sustainability scores and minimizing movement penalties. The model weights are saved as `dqn_material_env_weights.h5f`

### Video Demonstration

[Link to video](https://drive.google.com/file/d/1z-VfvpmQizMP1-UK2rWVK185Z8hG52Oi/view?usp=sharing)













