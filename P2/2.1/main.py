import yaml
import os
import neat
import numpy as np

from Entorno import Entorno
import Plots
from utils import evalua_genoma

with open("P2/2.1/config.yaml", "r") as file: config_global = yaml.safe_load(file)

entorno = Entorno(pasos_por_episodio=config_global['pasos_por_episodio'], alpha1=config_global['alpha1'],
                  alpha2=config_global['alpha2'], alpha3=config_global['alpha3'], sigma=config_global['sigma'])

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, 
                     neat.DefaultStagnation, 'P2/2.1/config.ini')

def evalua_genomas(genomes, config):
    for genoma_id, genoma in genomes:
        genoma.fitness = evalua_genoma(genoma, config, entorno)

poblacion = neat.Population(config)
winner = poblacion.run(evalua_genomas)
