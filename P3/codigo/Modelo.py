from utils import carga_politica, esta_viendo
from ModeloTelecontrol import carga_modelo_telecontrol

class Modelo:
    def __init__(self, ruta_politica, entorno, camara_telecontrol=None):
        self.modelo_telecontrol = carga_modelo_telecontrol()
        self.politica = carga_politica(ruta_politica, entorno)
        self.entorno = entorno
        self.camara_telecontrol = camara_telecontrol
        
        if camara_telecontrol is None:
            print("[Modelo] WARNING: No hay cámara de telecontrol disponible")

    def predict(self, frame_webcam, observacion):
        if esta_viendo(observacion):
            self.entorno.ui_origen = "POLITICA P1"
            return self.politica.predict(observacion)[0]
        
        else:
            self.entorno.ui_origen = "telecontrol"
            
            if self.camara_telecontrol is None or frame_webcam is None:
                import numpy as np
                print("[Modelo] Sin cámara de telecontrol - devolviendo acción nula")
                return np.array([0.0, 0.0], dtype=np.float32)
            
            return self.modelo_telecontrol.predict(frame_webcam)
