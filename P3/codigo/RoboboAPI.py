from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
import random
import numpy as np
import math

def init_Robobo(ip='localhost'):
    return Robobo(ip) 

def init_RoboboSim(ip='localhost'):
    return RoboboSim(ip)

def _get_xy(Entorno):
    """
    Metodo auxiliar, interfaz con robocop
    """
    blobs = Entorno.robocop.readAllColorBlobs()
    if blobs:
        for key in blobs:
            blob = blobs[key]
            return np.array([blob.posx,blob.posy])
    return np.array([-1,-1])


def _get_object_xz(Entorno):
    """
    Metodo auxiliar, interfaz con robocop
    """
    objetos = Entorno.sim.getObjects()
    if objetos != None and len(objetos) > 0:
        for objeto in objetos:
            posicion = Entorno.sim.getObjectLocation(objeto)['position']
            x_obj, z_obj = posicion['x'], posicion['z']
    
    return np.array([x_obj,z_obj])

        



def _get_robot_xz(Entorno):
    x_rob, y_rob = 0, 0
    posicion_robobo = Entorno.sim.getRobotLocation(0)['position']
    x_rob, z_rob = posicion_robobo['x'], posicion_robobo['z']
    return np.array([x_rob,z_rob])


def _get_IR(Entorno):
    """
    Metodo auxiliar, interfaz con robocop
    """
    irs = Entorno.robocop.readAllIRSensor()
    if irs != []:
        delante = irs["Front-C"]
        atras = irs["Back-C"]
        
        #print(f'delante {delante} atras {atras}')

        return np.array([delante, atras])
    else:
        return np.array([0, 0])

def _get_tamano_blob(Entorno):
    """
    Metodo auxiliar, interfaz con robocop
    """
    blobs = Entorno.robocop.readAllColorBlobs()
    if blobs:
        for key in blobs:
            blob = blobs[key]
            return np.array([blob.size])
    return np.array([-1])


def _distancia_a_blob(Entorno):
    x_obj, y_obj = 0, 0
    posicion_robobo = Entorno.sim.getRobotLocation(0)['position']
    x_rob, y_rob = posicion_robobo['x'], posicion_robobo['y']

    objetos = Entorno.sim.getObjects()
    if objetos != None and len(objetos) > 0:
        for objeto in objetos:
            posicion = Entorno.sim.getObjectLocation(objeto)['position']
            x_obj, y_obj = posicion['x'], posicion['y']

    #print(f'-> robobo ({x_rob},{y_rob}) -> objeto ({x_obj},{y_obj})')
    return math.sqrt((x_obj - x_rob)**2 + (y_obj - y_rob)**2)

def reset(Entorno):
    Entorno.sim.resetSimulation()
    Entorno.sim.wait(1)
    Entorno.robocop.moveTiltTo(110,100,wait=False)


def mover_blob_random_walk(entorno, dx, dz):
    """
    Mueve el blob en diagonal
    
    Args:
        entorno: El entorno de simulación de Robobo
        paso: Tamaño del paso en cada dirección (en metros)
    """
    # Obtener los objetos disponibles
    objetos = entorno.sim.getObjects()
    
    if objetos is None or len(objetos) == 0:
        print("No hay objetos en la simulación")
        return None
    
    # Usar el primer objeto
    if objetos != None and len(objetos) > 0:
        for objeto in objetos:
            posicion_actual = entorno.sim.getObjectLocation(objeto)['position']
    
    # Obtener posición actual del objeto
    x_actual = posicion_actual['x']
    y_actual = posicion_actual['y']
    z_actual = posicion_actual['z']
    

    if random.random() > 0.5:
        dx = dx
    else:
        dx = -dx
    if random.random() > 0.5:
        dz = dz
    else:
        dz = -dz
    
    # Movimiento diagonal (mismo paso en x e y)
    x_nueva = x_actual + dx
    z_nueva = z_actual + dz
    
    # Mover el objeto a la nueva posición
    entorno.sim.setObjectLocation(
        objeto, 
        position={'x': x_nueva, 'y': y_actual, 'z': z_nueva}
    )
    
    return None
