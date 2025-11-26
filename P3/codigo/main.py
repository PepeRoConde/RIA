import cv2
import yaml

from ModeloTelecontrol import carga_modelo_telecontrol
from camara import Camara
from utils import carga_politica
from Entorno import Entorno 
from Modelo import Modelo


with open("P3/configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)

camara = Camara()
entorno = Entorno(mundo_real=config['mundo_real'])
modelo = Modelo(config['ruta_politica'], entorno)

observacion, _  = entorno.reset()

while True:
    frame = camara.get_frame()
    if frame is None: continue
    accion = modelo.predict(frame, observacion) 
    observacion, recompensa, terminated, truncated, info = entorno.step(accion)

camara.stop()
cv2.destroyAllWindows()
