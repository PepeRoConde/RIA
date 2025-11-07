import yaml
import os
import neat
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from Entorno import Entorno
import Plots
from utils import evalua_genoma, guarda_genoma

with open("P2/configs/config.yaml", "r") as file: 
    config_global = yaml.safe_load(file)

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
mejor_fitness_por_generacion = []
fitness_promedio_por_generacion = []

def evalua_genomas(genomes, config):
    total_fitness = 0
    generacion_fitness = []
    
    with tqdm(total=len(genomes), desc=f"Generacion {len(generaciones)+1}", leave=True) as pbar:
        for i, (genoma_id, genoma) in enumerate(genomes, start=1):
            try:
                genoma.fitness = evalua_genoma(genoma, config, entorno)
            except Exception as e:
                print(f"\nError evaluando genoma {genoma_id}: {e}")
                genoma.fitness = -1000
            
            historial_fitness.append(genoma.fitness)
            generacion_fitness.append(genoma.fitness)
            total_fitness += genoma.fitness
            
            pbar.set_postfix(
                fitness=f"{genoma.fitness:.1f}", 
                avg=f"{total_fitness / i:.1f}", 
                best=f"{max(generacion_fitness):.1f}",
                worst=f"{min(generacion_fitness):.1f}"
            )
            pbar.update(1)
    
    generaciones.append(generacion_fitness)
    mejor_fitness_por_generacion.append(max(generacion_fitness))
    fitness_promedio_por_generacion.append(np.mean(generacion_fitness))
    
    print(f"\nGeneracion {len(generaciones)} - "
          f"Mejor: {max(generacion_fitness):.2f}, "
          f"Promedio: {np.mean(generacion_fitness):.2f}, "
          f"Std: {np.std(generacion_fitness):.2f}\n")


poblacion = neat.Population(config)
poblacion.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
poblacion.add_reporter(stats)

try:
    ganador = poblacion.run(evalua_genomas, n=config_global['num_generaciones'])
except KeyboardInterrupt:
    print("\nEntrenamiento interrumpido por el usuario")
    ganador = max(poblacion.population.values(), key=lambda g: g.fitness if g.fitness else -float('inf'))
    print(f"Usando mejor genoma encontrado con fitness={ganador.fitness:.2f}")

node_names = {
    -1: 'Blob_X',
    -2: 'Blob_Y', 
    -3: 'IR_delante',
    -4: 'IR_atras',
    -5: 'IR_ll',
    -6: 'IR_l',
    -7: 'IR_rr',
    -8: 'IR_r',
    -9: 'Tamano',
    0: 'Vel_Izq',
    1: 'Vel_Der'
}

print(f'\nGenoma ganador:')
print(f'  Fitness: {ganador.fitness:.2f}')
print(f'  Conexiones: {len(ganador.connections)}')
print(f'  Nodos: {len(ganador.nodes)}')

if config_global['guarda_genoma']: 
    nombre_archivo = guarda_genoma(ganador, config_global['genoma_archivo'])
else:
    nombre_archivo = ''

print("\nGenerando graficas...")

Plots.draw_net(config, ganador, True, 
               filename=f'P2/graficas/red_{nombre_archivo}', 
               node_names=node_names)

Plots.fitness_individuos(
    generaciones,
    nombre_figura=f"P2/graficas/individuos_{nombre_archivo}")

Plots.fitness_generaciones(
    generaciones,
    nombre_figura=f"P2/graficas/generaciones_{nombre_archivo}")

# Grafica de evolucion
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(mejor_fitness_por_generacion) + 1), 
         mejor_fitness_por_generacion, 'g-', linewidth=2, label='Mejor')
plt.plot(range(1, len(fitness_promedio_por_generacion) + 1), 
         fitness_promedio_por_generacion, 'b--', linewidth=2, label='Promedio')
plt.xlabel('Generacion')
plt.ylabel('Fitness')
plt.title('Evolucion del Fitness')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'P2/graficas/evolucion_{nombre_archivo}.png', dpi=150)
plt.close()

print(f"\nEntrenamiento completado:")
print(f"  Total generaciones: {len(generaciones)}")
print(f"  Mejor fitness: {max(mejor_fitness_por_generacion):.2f}")
print(f"  Genoma guardado: {nombre_archivo}")