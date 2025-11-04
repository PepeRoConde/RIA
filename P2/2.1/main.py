import yaml
import os
import neat

from Entorno import Entorno
import Plots
from utils import evalua_genomas

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                 'P2/2.1/config.ini')

print(type(config))

# Create the population, which is the top-level object for a NEAT run.
poblacion = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
poblacion.add_reporter(neat.StdOutReporter(False))

# Run until a solution is found.
winner = poblacion.run(evalua_genomas)
