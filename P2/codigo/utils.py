import numpy as np
import neat 
import pickle
import neat
import os

def guarda_genoma(genome, filename: str):

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    base, ext = os.path.splitext(filename)
    counter = 2
    new_filename = filename
    
    # Find a new filename if it already exists
    while os.path.exists(new_filename):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    
    with open(new_filename, "wb") as f:
        pickle.dump(genome, f)
    
    print(f"Genoma guardado en: {new_filename}")
    
    return os.path.splitext(os.path.basename(new_filename))[0]

def carga_genoma(filename: str, config: neat.Config):
    with open(filename, "rb") as f:
        genome = pickle.load(f)
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    print(f"Genoma cargado de: {filename}")
    
    return genome, net


def vectoriza_observacion(obs):
    return np.concatenate([
        obs["blob_xy"],
        obs["IR"],
        obs["tamano_blob"]
        
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
        
        # Escala de [-1, 1] (tanh) a [-15, 15]
        action = np.array(action_raw) * entorno.velocidad_max
        
        obs, recompensa, terminated, truncated, _ = entorno.step(action)
        obs_vector = vectoriza_observacion(obs)
        fitness += recompensa
        done = terminated or truncated
        paso += 1

    if entorno.verboso: print(f'Fitness {fitness}')
    return fitness
