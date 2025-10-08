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
                 sigma = 15):

        self.pasos_por_episodio =  pasos_por_episodio
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.sigma = sigma

        self.robocop = RoboboAPI.init_Robobo()
        self.robocop.connect()
        self.sim = RoboboAPI.init_RoboboSim()
        self.sim.connect()

        self.velocidad_min = -2 # se hara clip con estos valores
        self.velocidad_max = 2

        self.recompensas = []


        # inicializacion de las variables 
        self._blob_xy = np.array([-1, -1], dtype=np.int32)
        self._IR = np.array([0, 0], dtype=np.int32)
        self._tamano_blob = np.array([-1], dtype=np.int32)
        self._velocidad = np.array([0, 0], dtype=np.float32)      
        #self._distancia_a_blob = np.array([-1.], dtype=np.float32)

        # CAMBIAR: que sea 1000 me lo inventé
        tamano_blob_max = 1000
        IR_max = 10000
        
        # por un lado el xy que va de -1 a 101 y luego el tamano_blob
        self.observation_space = gym.spaces.Dict(
            {
                "blob_xy": gym.spaces.Box(-1, 102, shape=(2,), dtype=int),
                # ...size (int): The area of the blob measured in pixels.
                # es lo que pone en la documentacion de robobo sobre el tamano_blob
                "IR": gym.spaces.Box(0, IR_max, shape=(2,), dtype=int),
                "tamano_blob": gym.spaces.Box(0, tamano_blob_max, shape=(1,), dtype=int),
                "velocidad": gym.spaces.Box(self.velocidad_min, self.velocidad_max, shape=(2,), dtype=float)
                # pongo 1000 por poner
                #"distancia_a_blob": gym.spaces.Box(0,1000, shape=(1,), dtype=float)
            }
        )

        # va de 0 a 20 y en R2
        # si la accion es [2,4] no esque movewheels[2,4] 
        # esque movewheels[antesx + 2, antesy + 4]
        # asi mantener la velocidad es la misma accion independientemente del estado
        self.action_space = gym.spaces.Box(self.velocidad_min, self.velocidad_max, shape=(2,), dtype=float)

    def _get_observacion(self):
        """Convierte estado interno a observación

        Devuelve:
            dict: posicion de robobo y tamano del blob
        """
        #return {"blob_xy": self._blob_xy, "tamano_blob": self._tamano_blob, "distancia_a_blob": self._distancia_a_blob}
        #print(f'xy: {self._blob_xy}, tamano_blob: {self._tamano_blob}, velocidad {self._velocidad}')
        return {"blob_xy": self._blob_xy, 
                "IR": self._IR,
                "tamano_blob": self._tamano_blob, 
                "velocidad": self._velocidad}


    # todo lo relacionado con info es dummy pero por si luego queremos usarlo
    def _get_info(self):
        return {'supu':'tamadre'}

    def _get_xy(self):
        """
        Metodo auxiliar, interfaz con robocop
        """
        return RoboboAPI._get_xy(self)

    def _get_IR(self):
        """
        Metodo auxiliar, interfaz con robocop
        """
        return RoboboAPI._get_IR(self)

    def _get_tamano_blob(self):
        """
        Metodo auxiliar, interfaz con robocop
        """
        return RoboboAPI._get_tamano_blob(self)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Nuevo episodio

        Args:
            semilla: semilla para reproducibilidad

        Returns:
            tuple: (observation, info) for the initial state
        """
        # hola marce, esta linea tiene que ser así
        super().reset(seed=seed)

        print('=======RESET')

        RoboboAPI.reset(self)

        self.numero_de_pasos = 1


        # los metodos hablan con robocop y lo meten en las variables
        self._blob_xy = self._get_xy()
        self._IR = self._get_IR()
        self._tamano_blob = self._get_tamano_blob()

        # las variables son llamadas por este metodo que construye el diccionario
        observacion = self._get_observacion()
        info = self._get_info()
        
        #self.robocop.moveTiltTo(105,10)

        return observacion, info

    def _get_recompensa(self):
        """
        Método auxiliar para devolver la recompensa a partir de los atributos de la clase
        """

        x = self._blob_xy[0]
        d = RoboboAPI._distancia_a_blob(self)
        atras = self._IR[1]
        print(f'descentre: {(x-50)**2}, distancia_a_blob: {d}, atras: {max(0,atras-58)}, tamano_blob: {self._tamano_blob}')
        # OJO estoy usando el 50 pero si luego lo cambiamos a [0,1] habra que usar 0.5
        return self.alpha1 * math.exp(-(x-50)**2) + self.alpha2 * math.exp(-(d/self.sigma)**2) - self.alpha3 * max(0,atras-58) + 0.1 * float(self._tamano_blob)

    def step(self, accion):
        """Ejecuta un instante

        Args:
            accion: el incremento a tomar 

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        
        print(f'-- Paso #{self.numero_de_pasos}\n accion: {accion}')
        dx, dy = accion[0], accion[1]
        #print(f'dx: {dx}, dy: {dy}, velocidad: {self._velocidad}')
        self.robocop.moveWheels(self._velocidad[0] + dx, self._velocidad[1] + dy)
        time.sleep(1)
        self._velocidad[0] = np.clip(self._velocidad[0] + dx, self.velocidad_min, self.velocidad_max)
        self._velocidad[1] = np.clip(self._velocidad[1] + dy, self.velocidad_min, self.velocidad_max)

        print(f"VELOCIDAD {self._velocidad}")

        if self.numero_de_pasos == self.pasos_por_episodio:
            terminated, truncated  = True, True
        else:
            terminated, truncated = False, False

        self.numero_de_pasos += 1

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        recompensa = self._get_recompensa()
        self.recompensas.append(recompensa)
        print(f'recompensa: {recompensa}\n\n')
        
        self._blob_xy = self._get_xy()
        self._IR = self._get_IR()
        self._tamano_blob = self._get_tamano_blob()
        #self._distancia_a_blob = self._get_distancia_a_blob()

        observacion = self._get_observacion()
        info = self._get_info()

        return observacion, recompensa, terminated, truncated, info
