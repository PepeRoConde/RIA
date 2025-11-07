import yaml
import neat
import numpy as np

from Entorno import Entorno
from utils import vectoriza_observacion, carga_genoma 

with open("P2/configs/config.yaml", "r") as file: 
    config_global = yaml.safe_load(file)

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                     neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                     'P2/configs/config.ini')

entorno = Entorno(pasos_por_episodio=config_global['pasos_por_episodio'], 
                  alpha1=config_global['alpha1'],
                  alpha2=config_global['alpha2'], 
                  alpha3=config_global['alpha3'], 
                  verboso=config_global['verboso'],
                  sigma=config_global['sigma'],
                  velocidad_blob=config_global['velocidad_blob'],
                  posicion_inicial=config_global['posicion_inicial'])

# Cargar el genoma ganador
genoma, _ = carga_genoma(config_global['genoma_archivo'], config)
net = neat.nn.FeedForwardNetwork.create(genoma, config)

# Resetear el entorno
obs, _ = entorno.reset()
obs_vector = vectoriza_observacion(obs)

fitness = 0.0
done = False
paso = 0

print(f"Velocidades del entorno: min={entorno.velocidad_min}, max={entorno.velocidad_max}")
print(f"Primera observación: {obs_vector}\n")

while not done and paso < entorno.pasos_por_episodio:
    # Activar la red (salida en [0, 1] por sigmoid)
    action_raw = net.activate(obs_vector)
    
    # Escalar de [0, 1] a [velocidad_min, velocidad_max]
    action_raw = np.array(action_raw)
    action = action_raw * (entorno.velocidad_max - entorno.velocidad_min) + entorno.velocidad_min
    
    # Debug primeros pasos
    if paso < 3:
        print(f"Paso {paso}: action_sigmoid={action_raw}, action_escalada={action}")
    
    # Ejecutar acción
    obs, recompensa, terminated, truncated, _ = entorno.step(action)
    obs_vector = vectoriza_observacion(obs)
    fitness += recompensa
    done = terminated or truncated
    paso += 1

print(f'\nEl fitness en prueba, tras {paso} pasos es de: {fitness}')