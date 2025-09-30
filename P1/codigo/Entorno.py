from typing import Optional
import numpy as np
import gymnasium as gym
from robobopy.Robobo import Robobo
import math

# https://gymnasium.farama.org/introduction/create_custom_env/
# https://gymnasium.farama.org/api/spaces/
class Entorno(gym.Env):

    def __init__(self):


        self.robocop = Robobo("localhost")
        self.robocop.connect()


        # inicializacion de las variables 
        self._robobo_xy = np.array([-1, -1], dtype=np.int32)
        self._tamano_blob = np.array([-1], dtype=np.int32)
        #self._distancia_a_blob = np.array([-1.], dtype=np.float32)

        # CAMBIAR: que sea 100 me lo inventé
        tamano_blob_max = 100

        # por un lado el xy que va de -1 a 101 y luego el tamano_blob
        self.observation_space = gym.spaces.Dict(
            {
                "robobo_xy": gym.spaces.Box(-1, 102, shape=(2,), dtype=int),
                # ...size (int): The area of the blob measured in pixels.
                # es lo que pone en la documentacion de robobo sobre el tamano_blob
                "tamano_blob": gym.spaces.Box(0, tamano_blob_max, shape=(1,), dtype=int),

                # pongo 1000 por poner
                #"distancia_a_blob": gym.spaces.Box(0,1000, shape=(1,), dtype=float)
            }
        )

        # va de 0 a 20 y en R2
        # si la accion es [2,4] no esque movewheels[2,4] 
        # esque movewheels[antesx + 2, antesy + 4]
        # asi mantener la velocidad es la misma accion independientemente del estado
        self.action_space = gym.spaces.Box(-20, 20, shape=(2,), dtype=float)

    def _get_observacion(self):
        """Convierte estado interno a observación

        Devuelve:
            dict: posicion de robobo y tamano del blob
        """
        #return {"robobo_xy": self._robobo_xy, "tamano_blob": self._tamano_blob, "distancia_a_blob": self._distancia_a_blob}
        print(self._robobo_xy, self._tamano_blob)
        return {"robobo_xy": self._robobo_xy, "tamano_blob": self._tamano_blob}


    # todo lo relacionado con info es dummy pero por si luego queremos usarlo
    def _get_info(self):
        return {'supu':'tamadre'}

    def _get_xy(self):
        """
        Metodo auxiliar, interfaz con robocop
        """
        return np.array([0,0])

    def _get_tamano_blob(self):
        """
        Metodo auxiliar, interfaz con robocop
        """
        return np.array([0])

    #def _get_distancia_a_blob(self):
    #    """
    #    Metodo auxiliar, interfaz con robocop
    #    """
    #    pass

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Nuevo episodio

        Args:
            semilla: semilla para reproducibilidad

        Returns:
            tuple: (observation, info) for the initial state
        """
        # hola marce, esta linea tiene que ser así
        super().reset(seed=seed)

        # los metodos hablan con robocop y lo meten en las variables
        self._robobo_xy = self._get_xy()
        self._tamano_blob = self._get_tamano_blob()
        #self._distancia_a_blob = self._get_distancia_a_blob()

        # las variables son llamadas por este metodo que construye el diccionario
        observacion = self._get_observacion()
        info = self._get_info()

        return observacion, info

    def _get_recompensa(self):
        """
        Método auxiliar para devolver la recompensa a partir de los atributos de la clase
        """
        x = self._robobo_xy[0]
        #d = self._distancia_a_blob[0]
        alpha1 = 0.5
        #alpha2 = 0.5
        
        # OJO estoy usando el 50 pero si luego lo cambiamos a [0,1] habra que usar 0.5

        #return alpha1 * math.exp(-(x-50)**2) + alpha2 * math.exp(-(d)**2)
        return alpha1 * math.exp(-(x-50)**2)

    def step(self, accion):
        """Ejecuta un instante

        Args:
            accion: el incremento a tomar 

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-3) to a movement direction
        #direction = self._action_to_direction[action]

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        #self._agent_location = np.clip(
        #    self._agent_location + direction, 0, self.size - 1
        #)

        # Check if agent reached the target
        #terminated = np.array_equal(self._agent_location, self._target_location)
        terminated = False
        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        recompensa = self._get_recompensa()

        observacion = self._get_observacion()
        info = self._get_info()

        return observacion, recompensa, terminated, truncated, info
