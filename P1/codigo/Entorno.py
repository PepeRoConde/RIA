from typing import Optional
import time
import numpy as np
import gymnasium as gym
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
import math

class Entorno(gym.Env):

    def __init__(self):
        self.robocop = Robobo("localhost")
        self.robocop.connect()
        self.sim = RoboboSim("localhost")
        self.sim.connect()
        self.velocidad_min = -40
        self.velocidad_max = 40
        self.robocop.moveTiltTo(95,5)
        # inicializacion de las variables 
        self._blob_xy = np.array([-1, -1], dtype=np.int32)
        self._blob_xy_anterior = np.array([-1, -1], dtype=np.int32)  # NUEVO: para recordar última posición
        self._tamano_blob = np.array([-1], dtype=np.int32)
        self._velocidad = np.array([0, 0], dtype=np.float32)

        tamano_blob_max = 100

        self.observation_space = gym.spaces.Dict(
            {
                "blob_xy": gym.spaces.Box(-1, 102, shape=(2,), dtype=int),
                "tamano_blob": gym.spaces.Box(0, tamano_blob_max, shape=(1,), dtype=int),
                "velocidad": gym.spaces.Box(self.velocidad_min, self.velocidad_max, shape=(2,), dtype=float)
            }
        )

        self.action_space = gym.spaces.Box(self.velocidad_min, self.velocidad_max, shape=(2,), dtype=float)

    def _get_observacion(self):
        """Convierte estado interno a observación"""
        print(f'xy: {self._blob_xy}, tamano_blob: {self._tamano_blob}, velocidad {self._velocidad}')
        return {"blob_xy": self._blob_xy, 
                "tamano_blob": self._tamano_blob, 
                "velocidad": self._velocidad}

    def _get_info(self):
        return {'supu':'tamadre'}

    def _get_xy(self):
        """
        Metodo auxiliar, interfaz con robocop.
        Ahora también maneja la memoria de dirección cuando el blob desaparece.
        """
        blobs = self.robocop.readAllColorBlobs()
        
        if blobs:
            # El blob está visible
            for key in blobs:
                blob = blobs[key]
                nueva_posicion = np.array([blob.posx, blob.posy])
                # Guardamos la posición anterior antes de actualizar
                self._blob_xy_anterior = self._blob_xy.copy() if self._blob_xy[0] != -1 else nueva_posicion.copy()
                return nueva_posicion
        else:
            # El blob NO está visible
            # Comparamos con la última posición conocida para saber hacia dónde se fue
            if self._blob_xy_anterior[0] != -1:  # Si teníamos una posición anterior válida
                x_anterior = self._blob_xy_anterior[0]
                
                # Determinamos la dirección basándonos en la última posición conocida
                if x_anterior < 50:
                    # Estaba a la izquierda, probablemente se fue más a la izquierda
                    return np.array([-1, self._blob_xy_anterior[1]])
                else:
                    # Estaba a la derecha, probablemente se fue más a la derecha
                    return np.array([101, self._blob_xy_anterior[1]])
            else:
                # No hay información previa, devolvemos -1 por defecto
                return np.array([-1, -1])

    def _get_tamano_blob(self):
        """
        Metodo auxiliar, interfaz con robocop
        """
        blobs = self.robocop.readAllColorBlobs()
        if blobs:
            for key in blobs:
                blob = blobs[key]
                return np.array([blob.size])
        return np.array([-1])

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Nuevo episodio"""
        super().reset(seed=seed)

        # Reiniciamos también la memoria de posición anterior
        self._blob_xy_anterior = np.array([-1, -1], dtype=np.int32)
        
        self._blob_xy = self._get_xy()
        self._tamano_blob = self._get_tamano_blob()

        observacion = self._get_observacion()
        info = self._get_info()

        return observacion, info

    def _get_recompensa(self):
        """
        Método auxiliar para devolver la recompensa a partir de los atributos de la clase
        """
        def _distancia_a_blob():
            x_obj, y_obj = 0, 0
            posicion_robobo = self.sim.getRobotLocation(0)['position']
            x_rob, y_rob = posicion_robobo['x'], posicion_robobo['y']

            objetos = self.sim.getObjects()
            if objetos != None and len(objetos) > 0:
                for objeto in objetos:
                    posicion = self.sim.getObjectLocation(objeto)['position']
                    x_obj, y_obj = posicion['x'], posicion['y']

            print(f'-> robobo ({x_rob},{y_rob}) -> objeto ({x_obj},{y_obj})')
            return math.sqrt((x_obj - x_rob)**2 + (y_obj - y_rob)**2)

        x = self._blob_xy[0]
        d = _distancia_a_blob()
        
            
    
        if 0 <= x <= 100:
            descentrado = abs(x - 50)
            recompensa = -d - descentrado
        
        return recompensa
        
        

    def step(self, accion):
        """Ejecuta un instante"""
        # Ahora las acciones van de 0 a 20 directamente (sin conversión)
        dx = accion[0]  # 0 -> frenar, 10 -> velocidad media, 20 -> máxima
        dy = accion[1]
        
        print(f'accion: {accion}, velocidad_antes: {self._velocidad}')
        
        # Directamente usamos la acción como velocidad objetivo
        self._velocidad[0] = np.clip(dx, self.velocidad_min, self.velocidad_max)
        self._velocidad[1] = np.clip(dy, self.velocidad_min, self.velocidad_max)
        
        print(f'velocidad_despues: {self._velocidad}')
        
        # Enviamos la velocidad
        self.robocop.moveWheels(int(self._velocidad[0]), int(self._velocidad[1]))
        time.sleep(1)
        
        terminated = False
        truncated = False

        recompensa = self._get_recompensa()
        print(f'recompensa: {recompensa}\n\n')
        
        self._blob_xy = self._get_xy()
        self._tamano_blob = self._get_tamano_blob()

        observacion = self._get_observacion()
        info = self._get_info()

        return observacion, recompensa, terminated, truncated, info