import yaml
import os
import neat
import numpy as np

from Entorno import Entorno
import Plots
#from utils import evalua_genomas

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

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                 'P2/2.1/config.ini')

print(type(config))

poblacion = neat.Population(config)
#poblacion.add_reporter(neat.StdOutReporter(False))


def vectoriza_observacion(obs):
    return np.concatenate([
        obs["blob_xy"],
        obs["IR"],
        obs["tamano_blob"],
        obs["velocidad"]
    ])


def evalua_genomas(genomes, config):

    def vectoriza_observacion(obs):
        return np.concatenate([
            obs["blob_xy"],
            obs["IR"],
            obs["tamano_blob"],
            obs["velocidad"]
        ])

    def evalua_genoma(genoma, config):
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
    
    for genoma_id, genoma in genomes:
        print(f"Config type inside evalua_genomas: {type(config)}")
        genoma.fitness = evalua_genoma(genoma, config)



winner = poblacion.run(evalua_genomas)

