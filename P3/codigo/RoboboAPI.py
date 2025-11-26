from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from utils import muestra
import random
import numpy as np
import math
import cv2

def init_Robobo(ip='localhost'):
    return Robobo(ip) 

def init_RoboboSim(ip='localhost'):
    return RoboboSim(ip)

def _get_xy(Entorno):

    if Entorno.mundo_real:
        # Usar cámara para detectar el objeto
        frame = Entorno.camara.get_frame()
        if frame is None:
            return np.array([-1, -1])
        
        x, y, tamano = Entorno.sensor_objeto.detectar_objeto(frame)
        
        # Opcional: visualizar la detección para debugging
        if Entorno.visualizar_detecciones:
            frame = Entorno.sensor_objeto.visualizar_deteccion(frame, x, y, tamano)
            muestra(frame, 'Deteccion Objeto')
        
        return np.array([x, y])
    else:
        # Usar sensores del Robobo (simulación)
        blobs = Entorno.robocop.readAllColorBlobs()
        if blobs:
            for key in blobs:
                blob = blobs[key]
                return np.array([blob.posx, blob.posy])
        return np.array([-1, -1])


def _get_tamano_blob(Entorno):

    if Entorno.mundo_real:
        # Usar cámara
        frame = Entorno.camara.get_frame()
        if frame is None:
            return np.array([-1])
        
        _, _, tamano = Entorno.sensor_objeto.detectar_objeto(frame)
        return np.array([tamano])
    else:
        # Usar sensores del Robobo
        blobs = Entorno.robocop.readAllColorBlobs()
        if blobs:
            for key in blobs:
                blob = blobs[key]
                return np.array([blob.size])
        return np.array([-1])


def _get_object_xz(Entorno):
    """
    Obtiene la posición 3D del objeto (solo disponible en simulación)
    """
    if Entorno.mundo_real:
        # En mundo real no tenemos acceso a coordenadas 3D exactas
        return np.array([0.0, 0.0])
    
    objetos = Entorno.sim.getObjects()
    if objetos != None and len(objetos) > 0:
        for objeto in objetos:
            posicion = Entorno.sim.getObjectLocation(objeto)['position']
            x_obj, z_obj = posicion['x'], posicion['z']
            return np.array([x_obj, z_obj])
    
    return np.array([0.0, 0.0])


def _get_robot_xz(Entorno):
    """
    Obtiene la posición 3D del robot (solo disponible en simulación)
    """
    if Entorno.mundo_real:
        # En mundo real no tenemos acceso a coordenadas 3D exactas
        return np.array([0.0, 0.0])
    
    posicion_robobo = Entorno.sim.getRobotLocation(0)['position']
    x_rob, z_rob = posicion_robobo['x'], posicion_robobo['z']
    return np.array([x_rob, z_rob])


def _get_IR(Entorno):
    """
    Obtiene lecturas de los sensores IR (disponible en simulación y mundo real)
    """
    irs = Entorno.robocop.readAllIRSensor()
    if irs != []:
        delante = irs["Front-C"]
        atras = irs["Back-C"]
        return np.array([delante, atras])
    else:
        return np.array([0, 0])


def _distancia_a_blob(Entorno):
    """
    Calcula distancia al objeto.
    - En simulación: usa coordenadas 3D exactas
    - En mundo real: estima usando el tamaño del objeto en la imagen
    """
    if Entorno.mundo_real:
        # Estimación basada en el tamaño del objeto en la imagen
        # Asumiendo que conocemos el tamaño real del objeto
        tamano = _get_tamano_blob(Entorno)[0]
        
        if tamano <= 0:
            return 100.0  # Distancia grande si no se detecta
        
        # Fórmula aproximada: distancia inversamente proporcional al área
        # Estos parámetros deberían calibrarse para tu setup específico
        AREA_REFERENCIA = 5000  # Área cuando el objeto está a 1 metro
        DISTANCIA_REFERENCIA = 1.0
        
        distancia_estimada = DISTANCIA_REFERENCIA * np.sqrt(AREA_REFERENCIA / tamano)
        return distancia_estimada
    else:
        # Usar coordenadas 3D del simulador
        posicion_robobo = Entorno.sim.getRobotLocation(0)['position']
        x_rob, y_rob = posicion_robobo['x'], posicion_robobo['y']

        objetos = Entorno.sim.getObjects()
        if objetos != None and len(objetos) > 0:
            for objeto in objetos:
                posicion = Entorno.sim.getObjectLocation(objeto)['position']
                x_obj, y_obj = posicion['x'], posicion['y']
                return math.sqrt((x_obj - x_rob)**2 + (y_obj - y_rob)**2)
        
        return 100.0


def reset(Entorno):
    """
    Reinicia el entorno
    """
    if not Entorno.mundo_real:
        Entorno.sim.resetSimulation()
        Entorno.sim.wait(1)
    
    Entorno.robocop.moveTiltTo(110, 100, wait=False)


def mover_blob_random_walk(entorno, dx, dz):
    """
    Mueve el blob en diagonal (solo en simulación)
    """
    if entorno.mundo_real:
        # En mundo real el objeto no se mueve automáticamente
        return None
    
    objetos = entorno.sim.getObjects()
    
    if objetos is None or len(objetos) == 0:
        return None
    
    if objetos != None and len(objetos) > 0:
        for objeto in objetos:
            posicion_actual = entorno.sim.getObjectLocation(objeto)['position']
            
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
            
            x_nueva = x_actual + dx
            z_nueva = z_actual + dz
            
            entorno.sim.setObjectLocation(
                objeto, 
                position={'x': x_nueva, 'y': y_actual, 'z': z_nueva}
            )
    
    return None
