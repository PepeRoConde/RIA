import yaml 
import numpy as np
import cv2
from pathlib import Path

from utils import carga_modelo_YOLO, muestra

with open("P3/configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)


class ModeloTelecontrol:
    def __init__(self, velocidad_base=None):
        # Cargar sección de configuración de telecontrol
        config_tc = config.get('telecontrol', {})
        
        # Parámetros de velocidad
        self.velocidad_base = velocidad_base or config_tc.get('velocidad_base', 20)
        self.factor_normalizacion = config_tc.get('factor_normalizacion', 0.1)
        
        # Umbrales de detección de gestos
        config_gestos = config_tc.get('deteccion_gestos', {})
        self.umbral_brazo_arriba = config_gestos.get('umbral_brazo_arriba', 0.5)
        self.umbral_brazo_abajo = config_gestos.get('umbral_brazo_abajo', 1.0)
        self.umbral_manos_juntas = config_gestos.get('umbral_manos_juntas', 0.5)
        self.umbral_brazos_separados = config_gestos.get('umbral_brazos_separados', 1.5)
        self.umbral_manos_nivel = config_gestos.get('umbral_manos_nivel', 0.2)
        self.umbral_manos_pecho = config_gestos.get('umbral_manos_pecho', 0.3)
        
        # Ratios de movimiento
        config_movimiento = config_tc.get('ratios_movimiento', {})
        self.ratio_avance_giro = config_movimiento.get('ratio_avance_giro', 0.5)
        self.ratio_rotacion_giro = config_movimiento.get('ratio_rotacion_giro', 0.5)
        self.ratio_adelante = config_movimiento.get('ratio_adelante', 1.0)
        self.ratio_atras = config_movimiento.get('ratio_atras', 1.0)
        
        # Modelo YOLO
        self.YOLO = carga_modelo_YOLO()

    def _normalizar_velocidad(self, vel):
        """Convierte velocidad del rango [0, 20] al rango [-2, 2]"""
        return vel * self.factor_normalizacion
    
    def derecha(self, velocidad=None):
        """Gira a la derecha mientras avanza"""
        vel = velocidad or self.velocidad_base
        vel_norm = self._normalizar_velocidad(vel)
        
        avance_recto = vel_norm * self.ratio_avance_giro
        gire_derecha = vel_norm * self.ratio_rotacion_giro
        
        return np.array([avance_recto, gire_derecha], dtype=np.float32)
    
    def izquierda(self, velocidad=None):
        """Gira a la izquierda mientras avanza"""
        vel = velocidad or self.velocidad_base
        vel_norm = self._normalizar_velocidad(vel)
        
        avance_recto = vel_norm * self.ratio_avance_giro
        gire_derecha = -vel_norm * self.ratio_rotacion_giro
        
        return np.array([avance_recto, gire_derecha], dtype=np.float32)
    
    def adelante(self, velocidad=None):
        """Avanza recto"""
        vel = velocidad or self.velocidad_base
        vel_norm = self._normalizar_velocidad(vel)
        
        avance_recto = vel_norm * self.ratio_adelante
        gire_derecha = 0.0
        
        return np.array([avance_recto, gire_derecha], dtype=np.float32)
    
    def atras(self, velocidad=None):
        """Retrocede recto"""
        vel = velocidad or self.velocidad_base
        vel_norm = self._normalizar_velocidad(vel)
        
        avance_recto = -vel_norm * self.ratio_atras
        gire_derecha = 0.0
        
        return np.array([avance_recto, gire_derecha], dtype=np.float32)
    
    def quieto(self):
        """Detiene el movimiento"""
        return np.array([0.0, 0.0], dtype=np.float32)

    
    def detectar_posicion_brazos(self, keypoints):
        """
        Detecta la posición de los brazos basándose en keypoints.
        
        Todos los umbrales son configurables mediante config.yaml para facilitar
        el ajuste durante pruebas físicas.
        
        Returns:
            str: Nombre del gesto detectado
        """
        if len(keypoints) < 11:
            return "SIN BRAZOS"
    
        keypoints = np.array(keypoints)
        hombro_izq, hombro_der = keypoints[5], keypoints[6]
        codo_izq, codo_der = keypoints[7], keypoints[8]
        muneca_izq, muneca_der = keypoints[9], keypoints[10]
    
        # Verificación de detección
        brazos_detectados = {
            'izq': muneca_izq[0] > 0 and muneca_izq[1] > 0,
            'der': muneca_der[0] > 0 and muneca_der[1] > 0
        }
        if not brazos_detectados['izq'] and not brazos_detectados['der']:
            return "SIN BRAZOS"
    
        # Alturas relativas
        altura_hombro_izq, altura_hombro_der = hombro_izq[1], hombro_der[1]
        altura_muneca_izq, altura_muneca_der = muneca_izq[1], muneca_der[1]
        x_muneca_izq, x_muneca_der = muneca_izq[0], muneca_der[0]
    
        # Distancia entre hombros para normalizar
        ancho_hombros = abs(hombro_der[0] - hombro_izq[0])
        if ancho_hombros == 0:
            ancho_hombros = 1
    
        distancia_munecas = abs(x_muneca_der - x_muneca_izq)
    
        # Umbrales relativos (ahora configurables)
        brazo_izq_arriba = (altura_muneca_izq < 
                            altura_hombro_izq - self.umbral_brazo_arriba * ancho_hombros)
        brazo_der_arriba = (altura_muneca_der < 
                            altura_hombro_der - self.umbral_brazo_arriba * ancho_hombros)
        brazo_izq_abajo = (altura_muneca_izq > 
                           altura_hombro_izq + self.umbral_brazo_abajo * ancho_hombros)
        brazo_der_abajo = (altura_muneca_der > 
                           altura_hombro_der + self.umbral_brazo_abajo * ancho_hombros)
    
        # Detección de gestos
        if (brazo_izq_arriba and brazo_der_arriba and 
            distancia_munecas < self.umbral_manos_juntas * ancho_hombros):
            return "MANOS JUNTAS ARRIBA"
        
        if (brazo_izq_arriba and brazo_der_arriba and 
            distancia_munecas > self.umbral_brazos_separados * ancho_hombros):
            return "BRAZOS EN CRUZ"
        
        if brazo_izq_abajo and brazo_der_abajo:
            return "BRAZOS ABAJO"
        
        if brazo_der_arriba and not brazo_izq_arriba:
            return "BRAZO DERECHO"
        
        if brazo_izq_arriba and not brazo_der_arriba:
            return "BRAZO IZQUIERDO"
        
        if (distancia_munecas < self.umbral_manos_pecho * ancho_hombros and 
            abs(altura_muneca_izq - altura_muneca_der) < self.umbral_manos_nivel * ancho_hombros):
            return "MANOS JUNTAS PECHO"
        
        return "BRAZOS RELAJADOS"

    def predict(self, frame):
        """
        Predice la acción basándose en la detección de gestos.
        
        Args:
            frame: Frame de la cámara
            
        Returns:
            np.array: Acción [avance_recto, gire_derecha]
        """
        resultados = self.YOLO(frame, verbose=False)
        frame_anotado = resultados[0].plot()
        keypoint = resultados[0].keypoints.xy.cpu().numpy()[0]
        posicion = self.detectar_posicion_brazos(keypoint)
        muestra(frame_anotado, posicion) 

        match posicion:
            case "BRAZO DERECHO":
                return self.derecha()
        
            case "BRAZO IZQUIERDO":
                return self.izquierda()
        
            case "BRAZOS RELAJADOS" | "MANOS JUNTAS ARRIBA":
                return self.adelante()
        
            case "MANOS JUNTAS PECHO":
                return self.atras()
        
            case "BRAZOS EN CRUZ":
                return self.quieto()
            
            case _:
                return self.quieto()


def carga_modelo_telecontrol(velocidad_base=None):
    return ModeloTelecontrol(velocidad_base=velocidad_base)
