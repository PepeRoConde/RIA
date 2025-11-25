import cv2
import yaml

from acciones import get_acciones
from acciones_entorno import get_acciones_entorno
from vision import detectar_posicion_brazos
from camara import Camara
from utils import carga_politica
from Entorno import Entorno 

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

acciones = get_acciones()
ModeloTelecontrol = get_acciones_entorno() 
camara = Camara()
entorno = Entorno(pasos_por_episodio=pasos_por_episodio,
    alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, sigma=sigma)

politica_p1 = carga_politica(ruta_politica, entorno)


observacion, _  = entorno.reset()
for paso in range(pasos_por_episodio):
    frame = camara.get_frame()
    accion = ModeloTelecontrol.predict(frame) 
    observacion, recompensa, terminated, truncated, info = entorno.step(accion)


    
    print()

    

    '''
    resultados = modelo(frame)
    frame_anotado = resultados[0].plot()
    keypoint = resultados[0].keypoints.xy.cpu().numpy()[0]

    posicion = detectar_posicion_brazos(keypoint)
    print(f'posicion: {posicion} || accion: {acciones_ent.predict(posicion)}')
    cv2.putText(frame_anotado, f"Posición: {posicion}",
      (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
      (0, 255, 255), 3)

    cv2.imshow("YOLO - Control Robobo", frame_anotado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    '''
camara.stop()
cv2.destroyAllWindows()
