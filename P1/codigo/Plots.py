import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

def plot_recompensas(recompensas, pasos_por_episodio):
    plt.figure(figsize=(12, 5))
    plt.plot(recompensas, label='Reward per Step')

    for step in range(pasos_por_episodio, len(recompensas), pasos_por_episodio):
        plt.axvline(x=step, color='gray', linestyle='--', linewidth=0.5)

    plt.xlabel("Paso")
    plt.ylabel("Recompensa")
    #plt.title()
    plt.grid(True)
    #plt.legend()
    plt.show()

def plot_trayectorias(xy_objeto, xy_robot, title="Trayectorias del Robot y Objeto"):
    """
    Visualiza las trayectorias del robot y el objeto
    
    Args:
        xy_objeto: Lista o array con coordenadas del objeto
        xy_robot: Lista o array con coordenadas del robot
        title: Título del gráfico
    """
    
    xy_objeto = np.array(xy_objeto)
    xy_robot = np.array(xy_robot)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot de trayectorias
    if len(xy_objeto) > 0:
        ax.plot(xy_objeto[:, 0], xy_objeto[:, 1], 'b-o', 
                label='Objeto', linewidth=2, markersize=6, alpha=0.7)
        ax.plot(xy_objeto[0, 0], xy_objeto[0, 1], 'go', 
                markersize=12, label='Inicio Objeto', zorder=5)
        ax.plot(xy_objeto[-1, 0], xy_objeto[-1, 1], 'r*', 
                markersize=20, label='Fin Objeto', zorder=5)
    
    if len(xy_robot) > 0:
        ax.plot(xy_robot[:, 0], xy_robot[:, 1], 'r--s', 
                label='Robot', linewidth=2, markersize=6, alpha=0.7)
        ax.plot(xy_robot[0, 0], xy_robot[0, 1], 'cs', 
                markersize=12, label='Inicio Robot', zorder=5)
        ax.plot(xy_robot[-1, 0], xy_robot[-1, 1], 'mx', 
                markersize=12, label='Fin Robot', zorder=5)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()



def plot_recompensas_con_episodios(historial_recompensas, title="Recompensas por Step", suavizar=True, sigma=2):
    """
    Plotea las recompensas a lo largo del tiempo, marcando los límites entre episodios
    
    Args:
        historial_recompensas: Lista de listas con recompensas de episodios completados
        title: Título del gráfico
        suavizar: Si True, aplica suavizado gaussiano
        sigma: Desviación estándar de la gaussiana (mayor = más suave)
    """
    
    # Concatenar todos los episodios completados
    todas_las_recompensas = []
    limites_episodios = []
    
    for episodio in historial_recompensas:
        todas_las_recompensas.extend(episodio)
        limites_episodios.append(len(todas_las_recompensas))
    
    todas_las_recompensas = np.array(todas_las_recompensas)
    
    # Aplicar suavizado gaussiano si se solicita
    if suavizar and len(todas_las_recompensas) > 0:
        recompensas_suavizadas = gaussian_filter1d(todas_las_recompensas, sigma=sigma)
    else:
        recompensas_suavizadas = todas_las_recompensas
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot de recompensas
    steps = np.arange(len(todas_las_recompensas))
    
    # Plotear ambas: original (transparente) y suavizada
    if suavizar:
        ax.plot(steps, todas_las_recompensas, 'b-o', linewidth=1, markersize=2, 
                alpha=0.3, label='Recompensas originales')
        ax.plot(steps, recompensas_suavizadas, 'b-', linewidth=3, 
                alpha=0.8, label=f'Recompensas suavizadas (σ={sigma})')
    else:
        ax.plot(steps, todas_las_recompensas, 'b-o', linewidth=2, markersize=4, 
                alpha=0.7, label='Recompensas')
    
    # Marcar límites de episodios
    for limite in limites_episodios:
        ax.axvline(x=limite - 0.5, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax.text(limite - 0.5, ax.get_ylim()[1] * 0.95, '|', 
                fontsize=10, ha='center', color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Recompensa', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()





def plot_trayectorias_episodios(historial_xy_objeto, historial_xy_robot, title="Trayectorias concatenadas por episodio",name="name", separacion=2.0):
    """
    Plotea las trayectorias del objeto y del robot en el plano XY,
    desplazando cada episodio hacia la derecha para que no se superpongan.

    Args:
        historial_xy_objeto: Lista de listas con coordenadas XY del objeto por episodio
        historial_xy_robot: Lista de listas con coordenadas XY del robot por episodio
        title: Título del gráfico
        separacion: Distancia de desplazamiento horizontal entre episodios
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    offset = 0.0  # desplazamiento acumulado en el eje X
    
    for i, (xy_obj, xy_rob) in enumerate(zip(historial_xy_objeto, historial_xy_robot)):
        # Convertir a float para evitar errores al desplazar
        xy_obj = np.array(xy_obj, dtype=float)
        xy_rob = np.array(xy_rob, dtype=float)
        
        # Desplazar todo el episodio en X
        xy_obj_despl = xy_obj.copy()
        xy_rob_despl = xy_rob.copy()
        xy_obj_despl[:, 0] += offset
        xy_rob_despl[:, 0] += offset
        
        # Plotear trayectorias
        ax.plot(xy_obj_despl[:, 0], xy_obj_despl[:, 1], '-', color='orange', alpha=0.8, label='Objeto' if i == 0 else "")
        ax.plot(xy_rob_despl[:, 0], xy_rob_despl[:, 1], '-', color='blue', alpha=0.8, label='Robot' if i == 0 else "")
        
        # Marcar inicio y fin del episodio
        ax.scatter(xy_obj_despl[0, 0], xy_obj_despl[0, 1], color='orange', s=30, marker='o')
        ax.scatter(xy_rob_despl[0, 0], xy_rob_despl[0, 1], color='blue', s=30, marker='o')
        ax.scatter(xy_obj_despl[-1, 0], xy_obj_despl[-1, 1], color='orange', s=30, marker='x')
        ax.scatter(xy_rob_despl[-1, 0], xy_rob_despl[-1, 1], color='blue', s=30, marker='x')
        
        # Actualizar desplazamiento para el siguiente episodio
        rango_x = xy_obj[:, 0].max() - xy_obj[:, 0].min()
        offset += rango_x + separacion
    
    ax.set_xlabel("Posición X", fontsize=12)
    ax.set_ylabel("Posición Y", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    ruta_salida = f"P1/figures/{name}.png"
    fig.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    plt.show()

def plot_recompensas_episodios(historial_recompensas, title="Recompensas concatenadas por episodio",name = "name", separacion=5.0):
    """
    Plotea las recompensas de varios episodios, desplazando cada uno en el eje X
    para que no se superpongan visualmente.

    Args:
        historial_recompensas: Lista de listas, donde cada sublista contiene las recompensas por step en un episodio.
        title: Título del gráfico.
        separacion: Espacio horizontal entre episodios.
    """

    fig, ax = plt.subplots(figsize=(12, 6))
    offset = 0.0  # desplazamiento acumulado en el eje X

    for i, recompensas in enumerate(historial_recompensas):
        recompensas = np.array(recompensas, dtype=float)

        # Eje X desplazado (steps dentro del episodio + offset)
        x_vals = np.linspace(offset, offset + len(recompensas), len(recompensas))

        # Plotear la curva de recompensas
        ax.plot(x_vals, recompensas, label=f"Episodio {i+1}", alpha=0.8)

        # Marcar inicio y fin
        ax.scatter(x_vals[0], recompensas[0], color=ax.lines[-1].get_color(), s=30, marker='o')
        ax.scatter(x_vals[-1], recompensas[-1], color=ax.lines[-1].get_color(), s=30, marker='x')

        # Actualizar offset para el siguiente episodio
        offset += len(recompensas) + separacion

    # Etiquetas y formato
    ax.set_xlabel("Step (concatenado por episodios)", fontsize=12)
    ax.set_ylabel("Recompensa", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    ruta_salida = f"P1/figures/{name}.png"
    fig.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    plt.show()



def plot_ultimo_episodio(lista1, lista2):
    """
    Plotea el último episodio de cada lista
    
    Args:
        lista1: Lista de episodios (naranja)
        lista2: Lista de episodios (azul)
    """
    plt.figure(figsize=(12, 8))
    
    # Obtener último episodio de lista1
    if len(lista1) > 0 and len(lista1[-1]) > 0:
        ultimo_ep1 = np.array([coord for coord in lista1[-1]])
        plt.plot(ultimo_ep1[:, 0], ultimo_ep1[:, 1], 'o-', color='orange', 
                linewidth=2, markersize=5, label='Lista 1 (último episodio)')
    
    # Obtener último episodio de lista2
    if len(lista2) > 0 and len(lista2[-1]) > 0:
        ultimo_ep2 = np.array([coord for coord in lista2[-1]])
        plt.plot(ultimo_ep2[:, 0], ultimo_ep2[:, 1], 'o-', color='blue', 
                linewidth=2, markersize=5, label='Lista 2 (último episodio)')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Último episodio de cada lista')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.show()




def plot_recompensas_ultimo_episodio(historial_recompensas, title="Recompensas último episodio",name="name"):
    """
    Plotea la curva de recompensas del último episodio.

    Args:
        historial_recompensas: Lista de listas con recompensas por step en cada episodio.
        title: Título del gráfico.
    """
    
    # Obtener último episodio
    recompensas = np.array(historial_recompensas[-1], dtype=float)
    x_vals = np.arange(len(recompensas))

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plotear recompensas
    ax.plot(x_vals, recompensas, '-', color='green', alpha=0.8, label='Recompensa')

    # Marcar inicio y fin
    ax.scatter(x_vals[0], recompensas[0], color='green', s=40, marker='o', label='Inicio')
    ax.scatter(x_vals[-1], recompensas[-1], color='green', s=40, marker='x', label='Fin')

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Recompensa", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()


    ruta_salida = f"P1/figures/{name}.png"
    fig.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    plt.show()
