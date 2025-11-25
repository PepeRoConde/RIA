import cv2
import yaml

from ModeloTelecontrol import carga_modelo_telecontrol
from camara import Camara
from utils import carga_politica
from Entorno import Entorno 
from Modelo import Modelo

###

with open("P3/configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)

pasos_por_episodio = config['pasos_por_episodio']
alpha1 = config['alpha1']
alpha2 = config['alpha2']
alpha3 = config['alpha3']
alpha4 = config['alpha4']
sigma = config['sigma']
ruta_politica = config['ruta_politica']

###

camara = Camara()
entorno = Entorno(pasos_por_episodio=pasos_por_episodio,
    alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, sigma=sigma)
modelo = Modelo(ruta_politica, entorno)

observacion, _  = entorno.reset()

while True:
    frame = camara.get_frame()
    if frame is None: continue
    accion = modelo.predict(frame, observacion) 
    observacion, recompensa, terminated, truncated, info = entorno.step(accion)

camara.stop()
cv2.destroyAllWindows()
