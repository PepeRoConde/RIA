from utils import carga_politica, esta_viendo
from ModeloTelecontrol import carga_modelo_telecontrol
class Modelo:
    def __init__(self, ruta_politica, entorno):
        self.modelo_telecontrol = carga_modelo_telecontrol() 
        self.politica = carga_politica(ruta_politica, entorno)

    def predict(self, frame, observacion):
        if esta_viendo(observacion):
            print('politica')
            return self.politica.predict(observacion)[0]
        else:
            print('telecontrol')
            return self.modelo_telecontrol.predict(frame)
