from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from Entorno import Entorno

env = Entorno()
# It will check your custom environment and output additional warnings if needed
check_env(env)

print(env.__dict__)
# Define and Train the agent
#model = A2C("CnnPolicy", env).learn(total_timesteps=1000)
