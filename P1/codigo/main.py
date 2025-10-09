import yaml
import os
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from Entorno import Entorno
import Plots

# Load configuration
with open("P1/codigo/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Unpack config
pasos_por_episodio = config['pasos_por_episodio']
numero_episodios = config['numero_episodios']
politica = config['politica']
alpha1 = config['alpha1']
alpha2 = config['alpha2']
alpha3 = config['alpha3']
sigma = config['sigma']

load_weights = config['load_weights']
weights_load_path = config['weights_load_path']
weights_save_base_path = config.get('weights_save_base_path', "model_weights")

# Setup environment
entorno = Entorno(
    pasos_por_episodio=pasos_por_episodio,
    alpha1=alpha1,
    alpha2=alpha2,
    alpha3=alpha3,
    sigma=sigma
)
check_env(entorno)

# Setup model
if load_weights:
    modelo = SAC.load(weights_load_path, env=entorno)
    print(f"Loaded model from {weights_load_path}")
else:
    print('No se cargo modelo')
    modelo = SAC(politica, entorno)

# Train model
modelo.learn(total_timesteps=pasos_por_episodio * numero_episodios)

# Save model with hyperparameter-based path
save_name = f"sac_alpha1_{alpha1}_alpha2_{alpha2}_alpha3_{alpha3}_sigma_{sigma}_numeps{numero_episodios}.zip"
save_path = os.path.join(weights_save_base_path, save_name)
os.makedirs(weights_save_base_path, exist_ok=True)
modelo.save(save_path)
print(f"Model saved to {save_path}")


# Plot results
#Plots.plot_recompensas(entorno.recompensas, pasos_por_episodio)
#Plots.plot_trayectorias(entorno.xy_objeto, entorno.xy_robot)
Plots.plot_recompensas_episodios(entorno.historial_recompensas,name="histRecompensas")
Plots.plot_trayectorias_episodios(entorno.historial_xy_objeto, entorno.historial_xy_robot,name = "histCoord")
Plots.plot_recompensas_ultimo_episodio(entorno.historial_recompensas, name = "recompensas")
Plots.plot_trayectoria_ultimo_episodio(entorno.historial_xy_objeto, entorno.historial_xy_robot,name = "coords")
print(entorno.historial_xy_objeto)
