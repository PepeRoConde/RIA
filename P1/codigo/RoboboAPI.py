from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim

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

def _get_IR(Entorno):
    """
    Metodo auxiliar, interfaz con robocop
    """
    irs = Entorno.robocop.readAllIRSensor()
    if irs != []:
        delante = irs["Front-C"]
        atras = irs["Back-C"]
        
        print(f'delante {delante} atras {atras}')

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
