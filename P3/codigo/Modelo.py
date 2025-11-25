from utils import carga_politica, carga_modelo_telecontrol

class Modelo:
    def __init__(self, ruta_politica, entorno):
        self.modelo_telecontrol =  carga_modelo_telecontrol()
        self.politica = carga_politica(ruta_politica, entorno)

    def predict(self, observacion):
        if # lo esta vien
            return self.politica.predict(observacion)
        else:
            return self.modelo.predict(observacion)
