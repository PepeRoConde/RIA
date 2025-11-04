import numpy as np
import neat
import yaml

from Entorno import Entorno

def evalua_genomas(genomes, config):

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
        obs_vector = flatten_observation(obs)
    
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

        
    with open("P2/2.1/config.yaml", "r") as file: config = yaml.safe_load(file)
    
    pasos_por_episodio = config['pasos_por_episodio']
    alpha1 = config['alpha1']
    alpha2 = config['alpha2']
    alpha3 = config['alpha3']
    sigma = config['sigma']
    
    
    entorno = Entorno(
        pasos_por_episodio=pasos_por_episodio,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        sigma=sigma
    )

    
    for genoma_id, genoma in genomes:
        print(f"Config type inside evalua_genomas: {type(config)}")
        genoma.fitness = evalua_genoma(genoma, config, entorno)
