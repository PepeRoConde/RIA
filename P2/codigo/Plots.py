import matplotlib.pyplot as plt
import numpy as np
import graphviz
import os

def fitness_individuos(generaciones, nombre_figura=None):
    fitness = np.concatenate(generaciones)
    pasos_por_episodio = [len(g) for g in generaciones]  # population sizes per generation
    poblacion_promedio = int(np.mean(pasos_por_episodio))

    plt.figure(figsize=(12, 5))
    plt.plot(fitness, label='Fitness', color='blue')

    step = 0
    for n in pasos_por_episodio[:-1]:
        step += n
        plt.axvline(x=step, color='gray', linestyle='--', linewidth=0.5)

    plt.xlabel("Individuo (a lo largo de todas las generaciones)")
    plt.ylabel("Fitness")
    plt.title("Fitness de Individuos (por generación)")
    plt.legend()
    plt.grid(True)

    if nombre_figura:
        os.makedirs(os.path.dirname(nombre_figura), exist_ok=True)
        if not nombre_figura.endswith('.png'):
            nombre_figura += '.png'
        plt.savefig(nombre_figura, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Figura guardada en {nombre_figura}")
    else:
        plt.show()

def fitness_generaciones(generaciones, nombre_figura=None):
    mean_fit = [np.mean(g) for g in generaciones]
    max_fit = [np.max(g) for g in generaciones]
    min_fit = [np.min(g) for g in generaciones]
    std_fit = [np.std(g) for g in generaciones]

    gens = np.arange(1, len(generaciones) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(gens, mean_fit, label='Mean Fitness', color='blue', linewidth=2)
    plt.plot(gens, max_fit, label='Max Fitness', color='green', linestyle='--')
    plt.plot(gens, min_fit, label='Min Fitness', color='red', linestyle='--')

    plt.fill_between(gens,
                     np.array(mean_fit) - np.array(std_fit),
                     np.array(mean_fit) + np.array(std_fit),
                     color='blue', alpha=0.2, label='Mean ± 1 STD')

    plt.title("Fitness por Generación")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if nombre_figura:
        os.makedirs(os.path.dirname(nombre_figura), exist_ok=True)
        if not nombre_figura.endswith('.png'):
            nombre_figura += '.png'
        plt.savefig(nombre_figura, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Figura guardada en {nombre_figura}")
    else:
        plt.show()
# fusilado de https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/visualize.py
def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    # If requested, use a copy of the genome which omits all components that won't affect the output.
    if prune_unused:
        genome = genome.get_pruned_copy(config.genome_config)

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    used_nodes = set(genome.nodes.keys())
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot
