from typing import Optional
import time
import numpy as np
import gymnasium as gym
import math

import RoboboAPI

# https://gymnasium.farama.org/introduction/create_custom_env/
# https://gymnasium.farama.org/api/spaces/
class Entorno(gym.Env):

    def __init__(self, 
                pasos_por_episodio = 10,
                alpha1 = 0.5,
                alpha2 = 0.5,
                alpha3 = 0.00001,
                alpha4 = 0.1,
                sigma = 15,
                velocidad_blob = 20,
                mundo_real = False):

        self.pasos_por_episodio =  pasos_por_episodio
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4
        self.sigma = sigma
        self._velocidad_blob = velocidad_blob
        self.mundo_real = mundo_real

        self.robocop = RoboboAPI.init_Robobo()
        self.robocop.connect()
        self.sim = RoboboAPI.init_RoboboSim()
        self.sim.connect()

        self.velocidad_min = -2
        self.velocidad_max = 2

        # Historial total con sublistas por episodio
        self.historial_recompensas = []  # Lista de listas
        self.historial_xy_objeto = []    # Lista de listas
        self.historial_xy_robot = []     # Lista de listas
        
        # Variables para el episodio actual
        self.recompensas_episodio = []
        self.xy_objeto_episodio = []
        self.xy_robot_episodio = []

        # inicializacion de las variables 
        self._blob_xy = np.array([-1, -1], dtype=np.int32)
        self._IR = np.array([0, 0], dtype=np.int32)
        self._tamano_blob = np.array([-1], dtype=np.int32)
        self._velocidad = np.array([0, 0], dtype=np.float32)
        self.tamano_blob_max = 1000
        self.IR_max = 10000
        
        # por un lado el xy que va de -1 a 101 y luego el tamano_blob
        self.observation_space = gym.spaces.Dict(
            {
                "blob_xy": gym.spaces.Box(-1, 102, shape=(2,), dtype=int),
                "IR": gym.spaces.Box(0, self.tamano_blob_max, shape=(2,), dtype=int),
                "tamano_blob": gym.spaces.Box(0, 1000, shape=(1,), dtype=int),
                "velocidad": gym.spaces.Box(self.velocidad_min, self.velocidad_max, shape=(2,), dtype=float)
            }
        )

        # va de 0 a 20 y en R2
        # si la accion es [2,4] no esque movewheels[2,4] 
        # esque movewheels[antesx + 2, antesy + 4]
        # asi mantener la velocidad es la misma accion independientemente del estado
        self.action_space = gym.spaces.Box(self.velocidad_min, self.velocidad_max, shape=(2,), dtype=float)

    
    def _get_observacion(self):
        """Convierte estado interno a observación"""
        
        return {
            "blob_xy": self._blob_xy, 
            "IR": self._IR,
            "tamano_blob": self._tamano_blob, 
            "velocidad": self._velocidad
        }


    # todo lo relacionado con info es dummy pero por si luego queremos usarlo
    def _get_info(self):
        return {'supu':'tamadre'}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Nuevo episodio"""
        super().reset(seed=seed)

        print('=======RESET')

        # Guardar el episodio anterior en el historial total
        if self.recompensas_episodio:
            self.historial_recompensas.append(self.recompensas_episodio)
            self.historial_xy_objeto.append(self.xy_objeto_episodio)
            self.historial_xy_robot.append(self.xy_robot_episodio)

        # Limpiar listas del episodio actual
        self.recompensas_episodio = []
        self.xy_objeto_episodio = []
        self.xy_robot_episodio = []

        RoboboAPI.reset(self)

        self.numero_de_pasos = 1

        # los metodos hablan con robocop y lo meten en las variables
        self._blob_xy = RoboboAPI._get_xy(self)
        self._IR = RoboboAPI._get_IR(self)
        self._tamano_blob = RoboboAPI._get_tamano_blob(self)

        # Guardar posición inicial del objeto
        self.xy_objeto_episodio.append(RoboboAPI._get_object_xz(self))
        
        # Guardar posición inicial del robot
        robot_xy = RoboboAPI._get_robot_xz(self)  
        self.xy_robot_episodio.append(robot_xy)

        observacion = self._get_observacion()
        info = self._get_info()

        return observacion, info

    def _get_recompensa(self):
        """
        Método auxiliar para devolver la recompensa a partir de los atributos de la clase
        """

        x = self._blob_xy[0]
        d = RoboboAPI._distancia_a_blob(self)
        atras = self._IR[1]
        #print(f'descentre: {(x-50)**2}, distancia_a_blob: {d}, atras: {max(0,atras-58)}, tamano_blob: {self._tamano_blob}')
        return self.alpha1 * math.exp(-(x-50)**2) + self.alpha2 * math.exp(-(d/self.sigma)**2) - self.alpha3 * max(0,atras-58) + self.alpha4 * float(self._tamano_blob)

    def step(self, accion):
        """Ejecuta un instante"""
        
        print(f'-- Paso #{self.numero_de_pasos}\n accion: {accion}')
        avance_recto, gire_derecha = accion[0], accion[1]
        dx = avance_recto + gire_derecha
        dy = avance_recto - gire_derecha


        self.robocop.moveWheels(self._velocidad[0] + dx, self._velocidad[1] + dy)
        time.sleep(1)
        self._velocidad[0] = np.clip(self._velocidad[0] + dx, self.velocidad_min, self.velocidad_max)
        self._velocidad[1] = np.clip(self._velocidad[1] + dy, self.velocidad_min, self.velocidad_max)

        #print(f"VELOCIDAD {self._velocidad}")

        if self.numero_de_pasos == self.pasos_por_episodio:
            terminated, truncated = True, True
        else:
            terminated, truncated = False, False

        self.numero_de_pasos += 1

        recompensa = self._get_recompensa()
        #print(f'Recompensa: {recompensa}')
        self.recompensas_episodio.append(recompensa)
        #print(self.recompensas_episodio)
        
        
        self._blob_xy = RoboboAPI._get_xy(self)
        self._IR = RoboboAPI._get_IR(self)
        self._tamano_blob = RoboboAPI._get_tamano_blob(self)

        # Guardar posiciones en el historial del episodio
        self.xy_objeto_episodio.append(RoboboAPI._get_object_xz(self))
        #print(self.xy_objeto_episodio)
        
        robot_xy = RoboboAPI._get_robot_xz(self)  
        self.xy_robot_episodio.append(robot_xy)

        observacion = self._get_observacion()
        info = self._get_info()
        RoboboAPI.mover_blob_random_walk(self, self._velocidad_blob, self._velocidad_blob)
        return observacion, recompensa, terminated, truncated, info
