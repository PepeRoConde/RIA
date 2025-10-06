from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from Entorno import Entorno

env = Entorno()

# It will check your custom environment and output additional warnings if needed

# Define and Train the agent
model = A2C("MultiInputPolicy", env).learn(total_timesteps=100000)
