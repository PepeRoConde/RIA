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
generaciones = []

def evalua_genomas(genomes, config):
    total_fitness = 0
    generacion_fitnes = []
    with tqdm(total=len(genomes), desc="Evaluando genomas") as pbar:
        for i, (genoma_id, genoma) in enumerate(genomes, start=1):
            genoma.fitness = evalua_genoma(genoma, config, entorno)
            historial_fitness.append(genoma.fitness)
            generacion_fitnes.append(genoma.fitness)
            total_fitness += genoma.fitness
            pbar.set_postfix(fitness=f"{genoma.fitness:.2f}", avg=f"{total_fitness / i:.2f}", max=f"{max(historial_fitness):.2f}")
            pbar.update(1)
    generaciones.append(generacion_fitnes)



poblacion = neat.Population(config)
ganador = poblacion.run(evalua_genomas, n=config_global['num_generaciones'])
node_names=None # TODO: algo estilo  node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
print(f'--> El genoma ganador es:\n{ganador}')

if config_global['guarda_genoma']: 
    nombre_archivo = guarda_genoma(ganador, config_global['genoma_archivo'])
else:
    nombre_archivo = ''

Plots.draw_net(config, ganador, True, filename=f'P2/graficas/{nombre_archivo}', node_names=node_names)

Plots.fitness_individuos(
    generaciones,
    nombre_figura=f"P2/graficas/individuos_{nombre_archivo}")

Plots.fitness_generaciones(
    generaciones,
    nombre_figura=f"P2/graficas/generaciones_{nombre_archivo}")
