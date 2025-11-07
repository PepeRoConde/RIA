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
        self.velocidad_min = -30
        self.velocidad_max = 30

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
        self._IR = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)  # 6 sensores IR
        self._tamano_blob = np.array([-1], dtype=np.int32)
        self._velocidad = np.array([0, 0], dtype=np.float32)
        self.tamano_blob_max = 1000
        self.IR_max = 10000
        
        # Observación simplificada (sin velocidad)
        self.observation_space = gym.spaces.Dict(
            {
                "blob_xy": gym.spaces.Box(-1, 102, shape=(2,), dtype=int),
                "IR": gym.spaces.Box(0, self.IR_max, shape=(6,), dtype=int),  # 6 sensores
                "tamano_blob": gym.spaces.Box(0, 1000, shape=(1,), dtype=int),
            }
        )

        # Acción: velocidades directas de las ruedas [vel_izq, vel_der]
        self.action_space = gym.spaces.Box(self.velocidad_min, self.velocidad_max, shape=(2,), dtype=float)

    
    def _get_observacion(self):
        """Convierte estado interno a observación"""
        
        return {
            "blob_xy": self._blob_xy, 
            "IR": self._IR,
            "tamano_blob": self._tamano_blob
        }


    def _get_info(self):
        return {'velocidad': self._velocidad}

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

        # Resetear velocidades a 0
        self._velocidad = np.array([0, 0], dtype=np.float32)

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
        Recompensa enfocada en: centrar blob + acercarse usando IR + distancia + evitar paredes
        Lógica: Si ve el blob, ignora paredes (para perseguirlo). Si no lo ve, evita paredes (para buscarlo).
        """
        x = self._blob_xy[0]
        tamano = float(self._tamano_blob)
        atras = self._IR[1]
        delante = self._IR[0]
        d = RoboboAPI._distancia_a_blob(self)
        
        # Distancia (gaussiana)
        recompensa_distancia = 500 * math.exp(-(d / self.sigma) ** 2)
        
        ll = self._IR[2]  # Lateral izquierda exterior
        l = self._IR[3]   # Lateral izquierda interior
        rr = self._IR[4]  # Lateral derecha exterior
        r = self._IR[5]   # Lateral derecha interior
        
        # Detectar si ve el blob
        ve_blob = (x >0)
        
        # 1. CENTRADO (solo si ve el blob)
        if ve_blob:
            error_centrado = abs(x - 50)
            if error_centrado < 10:
                recompensa_centrado = 20
            elif error_centrado < 25:
                recompensa_centrado = 10
            else:
                recompensa_centrado = -5
        else:
            recompensa_centrado = 0
        
        # 2. TAMAÑO DEL BLOB (solo si lo ve)
        recompensa_tamano = tamano * 0.1 if ve_blob else 0
        
        # 3. IR DELANTERO (recompensa progresiva por acercarse)
        if delante > 500:  # Muy muy cerca del blob
            recompensa_IR = 100
        else:
            recompensa_IR = 0
        
        
        
        # 5. PENALIZACIONES POR PAREDES (solo si NO ve el blob)
        penalizacion_paredes = 0
        if not ve_blob:
            if ll > 70:
                penalizacion_paredes += (ll - 70) * 0.3
            if l > 70:
                penalizacion_paredes += (l - 70) * 0.5
            if rr > 70:
                penalizacion_paredes += (rr - 70) * 0.3
            if r > 70:
                penalizacion_paredes += (r - 70) * 0.5
            

        # 6. PENALIZACIÓN por perder el blob de vista
        penalizacion_perdido = -100 if not ve_blob else 0
        
        # 7. BONUS: si está centrado Y con IR alto = éxito
        bonus = 0
        if ve_blob and abs(x - 50) < 15 and delante > 300:
            bonus = 50
        
        # 8. INCENTIVO DE MOVIMIENTO si no ve blob (evita quedarse quieto)
        incentivo_movimiento = 0
        if not ve_blob:
            # Pequeña recompensa por moverse (basada en velocidad)
            velocidad_total = abs(self._velocidad[0]) + abs(self._velocidad[1])
            incentivo_movimiento = min(velocidad_total * 0.5, 10)  # Max 10
        
        if self.verboso:
            estado = "VE BLOB" if ve_blob else "NO VE BLOB"
            print(f'{estado} | blobx: {x:.1f}, centrado: {recompensa_centrado:.1f}, '
                f'distancia: {recompensa_distancia:.1f} (d={d:.1f}), '
                f'tamaño: {recompensa_tamano:.1f}, IR: {recompensa_IR:.1f}, '
                f'paredes: -{penalizacion_paredes:.1f}, perdido: {penalizacion_perdido}, '
                f'bonus: {bonus}, movimiento: {incentivo_movimiento:.1f}')
        
        recompensa = (
            recompensa_centrado +
            recompensa_tamano +
            recompensa_distancia +
            recompensa_IR +
            - penalizacion_paredes +
            penalizacion_perdido +
            bonus +
            incentivo_movimiento
        )
        
        return recompensa
    
    def step(self, accion):
        """
        Ejecuta un instante con velocidades directas
        """
        
        if self.verboso: print(f'-- Paso #{self.numero_de_pasos}\n accion: {accion}')
        
        # ⭐ VELOCIDADES DIRECTAS (no deltas)
        vel_izq, vel_der = accion[0], accion[1]
        
        # Clipea por seguridad
        vel_izq = np.clip(vel_izq, self.velocidad_min, self.velocidad_max)
        vel_der = np.clip(vel_der, self.velocidad_min, self.velocidad_max)
        
        # Actualiza el estado interno
        self._velocidad[0] = vel_izq
        self._velocidad[1] = vel_der
        
        # Mueve el robot
        self.robocop.moveWheels(vel_izq, vel_der)
        time.sleep(1)

        if self.verboso: print(f"VELOCIDAD [izq, der]: {self._velocidad}")

        # Chequea terminación
        if self.numero_de_pasos == self.pasos_por_episodio:
            terminated, truncated = True, True
        else:
            terminated, truncated = False, False

        self.numero_de_pasos += 1

        # Calcula recompensa
        recompensa = self._get_recompensa()
        if self.verboso: print(f'Recompensa: {recompensa}')
        self.recompensas_episodio.append(recompensa)
        
        # Actualiza sensores
        self._blob_xy = RoboboAPI._get_xy(self)
        self._IR = RoboboAPI._get_IR(self)
        self._tamano_blob = RoboboAPI._get_tamano_blob(self)

        # Guardar posiciones en el historial del episodio
        self.xy_objeto_episodio.append(RoboboAPI._get_object_xz(self))
        robot_xy = RoboboAPI._get_robot_xz(self)  
        self.xy_robot_episodio.append(robot_xy)

        observacion = self._get_observacion()
        info = self._get_info()
        
        # Mueve el blob
        RoboboAPI.mover_blob_random_walk(self, self._velocidad_blob, self._velocidad_blob)
        
        return observacion, recompensa, terminated, truncated, info