import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

from Entorno import Entorno
import Plots

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

pasos_por_episodio = config['pasos_por_episodio']
numero_episodios = config['numero_episodios']
politica = config['politica']
alpha1=config['alpha1']
alpha2=config['alpha2']
sigma=config['sigma']

# Setup environment and model
entorno = Entorno(pasos_por_episodio=pasos_por_episodio,
                  alpha1=alpha1,
                  alpha2=alpha2,
                  sigma=sigma)
check_env(entorno)

model = SAC(politica, entorno).learn(
    total_timesteps=pasos_por_episodio * numero_episodios
)

Plots.plot_recompensas(entorno.recompensas, pasos_por_episodio)

