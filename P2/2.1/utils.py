import numpy as np
import neat 

def vectoriza_observacion(obs):
    return np.concatenate([
        obs["blob_xy"],
        obs["IR"],
        obs["tamano_blob"],
        obs["velocidad"]
    ])

def evalua_genoma(genoma, config, entorno):
    net = neat.nn.FeedForwardNetwork.create(genoma, config)
    obs, _ = entorno.reset()
    obs_vector = vectoriza_observacion(obs)
    fitness = 0.0
    done = False
    paso = 0

    while not done and paso < entorno.pasos_por_episodio:
        action_raw = net.activate(obs_vector)
        action = np.clip(action_raw, entorno.velocidad_min, entorno.velocidad_max)
        obs, recompensa, terminated, truncated, _ = entorno.step(action)
        obs_vector = vectoriza_observacion(obs)
        fitness += recompensa
        done = terminated or truncated
        paso += 1

    print(f'Fitness {fitness}')
    return fitness
