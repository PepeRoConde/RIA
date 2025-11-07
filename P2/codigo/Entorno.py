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
                alpha5 = 3,
                alpha6 = 5,
                sigma = 15,
                verboso = False,
                velocidad_blob = 20,
                posicion_inicial = None):

        self.pasos_por_episodio =  pasos_por_episodio
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4
        self.alpha5 = alpha5
        self.alpha6 = alpha6
        self.sigma = sigma
        self.verboso = verboso
        self._velocidad_blob = velocidad_blob
        self._posicion_inicial = posicion_inicial

        self.robocop = RoboboAPI.init_Robobo()
        self.robocop.connect()
        self.sim = RoboboAPI.init_RoboboSim()
        self.sim.connect()
        self.sim.wait(1)
        self.velocidad_min = -15
        self.velocidad_max = 15

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

        
        if self.verboso: print('=======RESET')

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

        if self._posicion_inicial:
            print(self._posicion_inicial)
            RoboboAPI.mover_robobo_a_posicion(self, self._posicion_inicial['x'], self._posicion_inicial['y'], self._posicion_inicial['z'])


        return observacion, info

    def _get_recompensa(self):
        """
        Recompensa enfocada en: centrar blob + acercarse usando IR + evitar colisiones
        """
        x = self._blob_xy[0]
        tamano = float(self._tamano_blob)
        atras = self._IR[1]
        delante = self._IR[0]
        ll = self._IR[2]  # Lateral izquierda exterior
        l = self._IR[3]   # Lateral izquierda interior
        rr = self._IR[4]  # Lateral derecha exterior
        r = self._IR[5] 
        # 1. CENTRADO (crítico para mantener el blob en vista)
        error_centrado = abs(x - 50)
        if error_centrado < 10:
            recompensa_centrado = 20
        elif error_centrado < 25:
            recompensa_centrado = 10
        else:
            recompensa_centrado = -5
        
        # 2. TAMAÑO DEL BLOB (indica proximidad visual)
        recompensa_tamano = tamano * 0.1
        
        # 3. IR DELANTERO (recompensa progresiva por acercarse)
        if delante > 500:  # Muy muy cerca del blob
            recompensa_IR = 100
        elif delante > 300:  # Muy cerca
            recompensa_IR = 50
        elif delante > 200:  # Cerca
            recompensa_IR = 25
        elif delante > 100:  # Detectando algo adelante
            recompensa_IR = 10
        else:  # Nada adelante
            recompensa_IR = 0
        
        # 4. PENALIZACIÓN POR COLISIÓN TRASERA
        penalizacion_atras = -20 if atras > 100 else 0
        

        penalizacion_paredes = 0
    
        if ll > 100:  # Pared lateral izquierda externa cercana
            penalizacion_paredes += (ll - 100) * 0.15
        if l > 100:   # Pared lateral izquierda interna cercana (más crítica)
            penalizacion_paredes += (l - 100) * 0.25
        if rr > 100:  # Pared lateral derecha externa cercana
            penalizacion_paredes += (rr - 100) * 0.15
        if r > 100:   # Pared lateral derecha interna cercana (más crítica)
            penalizacion_paredes += (r - 100) * 0.25
        if atras > 100:  # Pared trasera
            penalizacion_paredes += (atras - 100) * 0.3

        # 5. PENALIZACIÓN por perder el blob de vista
        penalizacion_perdido = -100 if x == -1 else 0
        
        # 6. BONUS : si está centrado Y con IR alto = éxito
        bonus = 0
        if error_centrado < 15 and delante > 300:
            bonus = 50  
        
        if self.verboso:
            print(f'centrado: {recompensa_centrado:.1f}, tamaño: {recompensa_tamano:.1f}, '
                f'IR_delante: {recompensa_IR:.1f} (valor={delante}), atras: {penalizacion_atras}, '
                f'perdido: {penalizacion_perdido}, combo: {bonus}')
        
        recompensa = (
            recompensa_centrado +
            recompensa_tamano +
            recompensa_IR +
            penalizacion_atras +
            penalizacion_perdido +
            bonus
        )
        
        return recompensa
    def step(self, accion):
        """Ejecuta un instante"""
        
        if self.verboso: print(f'-- Paso #{self.numero_de_pasos}\n accion: {accion}')
        avance_recto, gire_derecha = accion[0], accion[1]
        dx = avance_recto + gire_derecha
        dy = avance_recto - gire_derecha


        self.robocop.moveWheels(self._velocidad[0] + dx, self._velocidad[1] + dy)
        time.sleep(1)
        self._velocidad[0] = np.clip(self._velocidad[0] + dx, self.velocidad_min, self.velocidad_max)
        self._velocidad[1] = np.clip(self._velocidad[1] + dy, self.velocidad_min, self.velocidad_max)

        if self.verboso: print(f"VELOCIDAD {self._velocidad}")

        if self.numero_de_pasos == self.pasos_por_episodio:
            terminated, truncated = True, True
        else:
            terminated, truncated = False, False

        self.numero_de_pasos += 1

        recompensa = self._get_recompensa()
        if self.verboso: print(f'Recompensa: {recompensa}')
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
