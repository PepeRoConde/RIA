from utils import carga_politica, carga_modelo

class Modelo:
    def __init__(self, ruta_politica, entorno):
        self.modelo =  carga_modelo()
        self.politica = carga_politica(ruta_politica, entorno)

    def predict(self, observacion):
        if # lo esta vien
            return self.politica.predict(observacion)
        else:
            return self.modelo.predict(observacion)
