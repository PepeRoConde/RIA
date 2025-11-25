import yaml 
import numpy as np
from vision import detectar_posicion_brazos 
import cv2

from utils import carga_modelo_YOLO

with open("P3/configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)


class ModeloTelecontrol:
    """
    Convierte acciones hardcoded a arrays compatibles con Entorno.step()
    
    El espacio de acciones de Entorno es Box[-2, 2] con shape=(2,)
    donde accion = [avance_recto, gire_derecha]
    
    En step():
    - dx = avance_recto + gire_derecha  (rueda izquierda)
    - dy = avance_recto - gire_derecha  (rueda derecha)
    """
    
    def __init__(self, velocidad_base=None):
        # Usa velocidad del config si no se especifica
        self.velocidad_base = velocidad_base or config['velocidad']
        
        # Normalizar velocidad al rango [-2, 2] del action_space
        # Asumiendo que velocidad en config es 0-20, normalizamos a -2, 2
        self.factor_normalizacion = 2.0 / 20.0
        self.YOLO = carga_modelo_YOLO()

    def _normalizar_velocidad(self, vel):
        """Convierte velocidad del rango del robot al rango del entorno"""
        return vel * self.factor_normalizacion
    
    def derecha(self, velocidad=None):
        """
        Girar a la derecha: rueda izquierda avanza, derecha quieta
        moveWheelsByTime(0, velocidad, tiempo) → queremos dy=0, dx=velocidad
        
        dx = avance_recto + gire_derecha = velocidad
        dy = avance_recto - gire_derecha = 0
        
        → avance_recto = velocidad/2, gire_derecha = velocidad/2
        """
        vel = velocidad or self.velocidad_base
        vel_norm = self._normalizar_velocidad(vel)
        
        avance_recto = vel_norm / 2
        gire_derecha = vel_norm / 2
        
        return np.array([avance_recto, gire_derecha], dtype=np.float32)
    
    def izquierda(self, velocidad=None):
        """
        Girar a la izquierda: rueda izquierda quieta, derecha avanza
        moveWheelsByTime(velocidad, 0, tiempo) → queremos dx=0, dy=velocidad
        
        dx = avance_recto + gire_derecha = 0
        dy = avance_recto - gire_derecha = velocidad
        
        → avance_recto = velocidad/2, gire_derecha = -velocidad/2
        """
        vel = velocidad or self.velocidad_base
        vel_norm = self._normalizar_velocidad(vel)
        
        avance_recto = vel_norm / 2
        gire_derecha = -vel_norm / 2
        
        return np.array([avance_recto, gire_derecha], dtype=np.float32)
    
    def adelante(self, velocidad=None):
        """
        Avanzar recto: ambas ruedas a la misma velocidad
        moveWheelsByTime(velocidad, velocidad, tiempo) → dx=dy=velocidad
        
        dx = avance_recto + gire_derecha = velocidad
        dy = avance_recto - gire_derecha = velocidad
        
        → avance_recto = velocidad, gire_derecha = 0
        """
        vel = velocidad or self.velocidad_base
        vel_norm = self._normalizar_velocidad(vel)
        
        avance_recto = vel_norm
        gire_derecha = 0.0
        
        return np.array([avance_recto, gire_derecha], dtype=np.float32)
    
    def atras(self, velocidad=None):
        """
        Retroceder: ambas ruedas a velocidad negativa
        moveWheelsByTime(-velocidad, -velocidad, tiempo) → dx=dy=-velocidad
        
        → avance_recto = -velocidad, gire_derecha = 0
        """
        vel = velocidad or self.velocidad_base
        vel_norm = self._normalizar_velocidad(vel)
        
        avance_recto = -vel_norm
        gire_derecha = 0.0
        
        return np.array([avance_recto, gire_derecha], dtype=np.float32)
    
    def quieto(self):
        """
        Detener: no aplicar ningún cambio de velocidad
        """
        return np.array([0.0, 0.0], dtype=np.float32)

    def predict(self, frame):


        resultados = self.YOLO(frame)
        frame_anotado = resultados[0].plot()
        keypoint = resultados[0].keypoints.xy.cpu().numpy()[0]

        posicion = detectar_posicion_brazos(keypoint)
        #print(f'posicion: {posicion} || accion: {acciones_ent.predict(posicion)}')

        
        cv2.putText(frame_anotado, f"Posición: {posicion}",
          (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
          (0, 255, 255), 3)

        cv2.imshow("YOLO - Control Robobo", frame_anotado)

        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

        if posicion == "BRAZO DERECHO":
            return self.derecha()
    
        elif posicion == "BRAZO IZQUIERDO":
            return self.izquierda()
    
        elif posicion == "BRAZOS RELAJADOS" or posicion == "MANOS JUNTAS ARRIBA":
            return self.adelante()
    
        elif posicion == "MANOS JUNTAS PECHO":
            return self.atras()
    
        elif posicion in ("BRAZOS EN CRUZ"):
            return self.quieto()

def get_acciones_entorno(velocidad_base=None):
    """
    Factory function para crear instancia de AccionesEntorno
    
    Args:
        velocidad_base: Velocidad base opcional. Si no se proporciona, usa config
    
    Returns:
        AccionesEntorno: Instancia lista para usar
    """

    return ModeloTelecontrol(velocidad_base=velocidad_base)
