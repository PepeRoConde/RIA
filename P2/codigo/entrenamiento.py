import yaml
import os
import neat
import numpy as np
from tqdm import tqdm

from Entorno import Entorno
import Plots
from utils import evalua_genoma, guarda_genoma

with open("P2/configs/config.yaml", "r") as file: config_global = yaml.safe_load(file)

entorno = Entorno(pasos_por_episodio=config_global['pasos_por_episodio'], 
                  alpha1=config_global['alpha1'],
                  alpha2=config_global['alpha2'], 
                  alpha3=config_global['alpha3'], 
                  verboso=config_global['verboso'],
                  sigma=config_global['sigma'],
                  velocidad_blob=config_global['velocidad_blob'],
                  posicion_inicial=config_global['posicion_inicial'])


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, 
                     neat.DefaultStagnation, 'P2/configs/config.ini')

historial_fitness = []

def evalua_genomas(genomes, config):
    total_fitness = 0
    with tqdm(total=len(genomes), desc="Evaluando genomas") as pbar:
        for i, (genoma_id, genoma) in enumerate(genomes, start=1):
            genoma.fitness = evalua_genoma(genoma, config, entorno)
            historial_fitness.append(genoma.fitness)
            total_fitness += genoma.fitness
            pbar.set_postfix(fitness=f"{genoma.fitness:.2f}", avg=f"{total_fitness / i:.2f}", max=f"{max(historial_fitness):.2f}")
            pbar.update(1)



poblacion = neat.Population(config)
ganador = poblacion.run(evalua_genomas)
Plots.fitness(historial_fitness, config_global['pasos_por_episodio'])
node_names=None # TODO: algo estilo  node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
Plots.draw_net(config, ganador, True, filename='P2/graficas/red', node_names=node_names)
print(f'--> El genoma ganador es:\n{ganador}')

if config_global['guarda_genoma']: guarda_genoma(ganador, config_global['genoma_archivo'])
