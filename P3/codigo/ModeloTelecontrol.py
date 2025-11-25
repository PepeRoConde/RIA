import yaml 
import numpy as np
import cv2

from utils import carga_modelo_YOLO, muestra

with open("P3/configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)


class ModeloTelecontrol:
    def __init__(self, velocidad_base=None):
        self.velocidad_base = velocidad_base or config['velocidad']
        self.factor_normalizacion = 2.0 / 20.0
        self.YOLO = carga_modelo_YOLO()

    def _normalizar_velocidad(self, vel):
        return vel * self.factor_normalizacion
    
    def derecha(self, velocidad=None):
        vel = velocidad or self.velocidad_base
        vel_norm = self._normalizar_velocidad(vel)
        
        avance_recto = vel_norm / 2
        gire_derecha = vel_norm / 2
        
        return np.array([avance_recto, gire_derecha], dtype=np.float32)
    
    def izquierda(self, velocidad=None):
        vel = velocidad or self.velocidad_base
        vel_norm = self._normalizar_velocidad(vel)
        
        avance_recto = vel_norm / 2
        gire_derecha = -vel_norm / 2
        
        return np.array([avance_recto, gire_derecha], dtype=np.float32)
    
    def adelante(self, velocidad=None):
        vel = velocidad or self.velocidad_base
        vel_norm = self._normalizar_velocidad(vel)
        
        avance_recto = vel_norm
        gire_derecha = 0.0
        
        return np.array([avance_recto, gire_derecha], dtype=np.float32)
    
    def atras(self, velocidad=None):
        vel = velocidad or self.velocidad_base
        vel_norm = self._normalizar_velocidad(vel)
        
        avance_recto = -vel_norm
        gire_derecha = 0.0
        
        return np.array([avance_recto, gire_derecha], dtype=np.float32)
    
    def quieto(self):
        return np.array([0.0, 0.0], dtype=np.float32)

    
    def detectar_posicion_brazos(self, keypoints):
        if len(keypoints) < 11:
            return "SIN BRAZOS"
    
        keypoints = np.array(keypoints)
        hombro_izq, hombro_der = keypoints[5], keypoints[6]
        codo_izq, codo_der = keypoints[7], keypoints[8]
        muñeca_izq, muñeca_der = keypoints[9], keypoints[10]
    
        # Verificación de detección
        brazos_detectados = {
            'izq': muñeca_izq[0] > 0 and muñeca_izq[1] > 0,
            'der': muñeca_der[0] > 0 and muñeca_der[1] > 0
        }
        if not brazos_detectados['izq'] and not brazos_detectados['der']:
            return "SIN BRAZOS"
    
        # Alturas relativas
        altura_hombro_izq, altura_hombro_der = hombro_izq[1], hombro_der[1]
        altura_muñeca_izq, altura_muñeca_der = muñeca_izq[1], muñeca_der[1]
        x_muñeca_izq, x_muñeca_der = muñeca_izq[0], muñeca_der[0]
    
        # Distancia entre hombros para normalizar
        ancho_hombros = abs(hombro_der[0] - hombro_izq[0])
        if ancho_hombros == 0:
            ancho_hombros = 1
    
        distancia_muñecas = abs(x_muñeca_der - x_muñeca_izq)
    
        # Umbrales relativos
        brazo_izq_arriba = altura_muñeca_izq < altura_hombro_izq - 0.5 * ancho_hombros
        brazo_der_arriba = altura_muñeca_der < altura_hombro_der - 0.5 * ancho_hombros
        brazo_izq_abajo = altura_muñeca_izq > altura_hombro_izq + 1.0 * ancho_hombros
        brazo_der_abajo = altura_muñeca_der > altura_hombro_der + 1.0 * ancho_hombros
    
        # Nuevos gestos
        if brazo_izq_arriba and brazo_der_arriba and distancia_muñecas < 0.5 * ancho_hombros:
            return "MANOS JUNTAS ARRIBA"
        if brazo_izq_arriba and brazo_der_arriba and distancia_muñecas > 1.5 * ancho_hombros:
            return "BRAZOS EN CRUZ"
        if brazo_izq_abajo and brazo_der_abajo:
            return "BRAZOS ABAJO"
        if brazo_der_arriba and not brazo_izq_arriba:
            return "BRAZO DERECHO"
        if brazo_izq_arriba and not brazo_der_arriba:
            return "BRAZO IZQUIERDO"
        if distancia_muñecas < 0.3 * ancho_hombros and abs(altura_muñeca_izq - altura_muñeca_der) < 0.2 * ancho_hombros:
            return "MANOS JUNTAS PECHO"
        
        return "BRAZOS RELAJADOS"

    def predict(self, frame):


        resultados = self.YOLO(frame)
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

def carga_modelo_telecontrol(velocidad_base=None):
    return ModeloTelecontrol(velocidad_base=velocidad_base)
