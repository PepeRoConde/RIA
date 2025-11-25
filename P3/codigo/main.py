import cv2
import yaml

from acciones import get_acciones
from acciones_entorno import get_acciones_entorno
from vision import detectar_posicion_brazos
from camara import Camara
from utils import carga_politica, carga_modelo
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
acciones_ent = get_acciones_entorno() 
modelo = carga_modelo()
camara = Camara()
entorno = Entorno(pasos_por_episodio=pasos_por_episodio,
    alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, sigma=sigma)

politica_p1 = carga_politica(ruta_politica, entorno)

ha_visto_blob = False

print(f'{acciones_ent.derecha()} ')
print("Presiona 'q' para salir")

while not ha_visto_blob:

    frame = cv2.resize(cv2.flip(camara.get_frame(), 1), (640, 480))
    resultados = modelo(frame)
    frame_anotado = resultados[0].plot()
    print('len ',len(resultados))
    for resultado in resultados:
        if resultado.keypoints is not None:
            keypoints = resultado.keypoints.xy.cpu().numpy()

            for person_keypoints in keypoints:

                posicion = detectar_posicion_brazos(person_keypoints)
                print(f'posicion: {posicion} || accion: {acciones_ent.predict(posicion)}')
                cv2.putText(frame_anotado, f"Posición: {posicion}",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                            (0, 255, 255), 3)

    cv2.imshow("YOLO - Control Robobo", frame_anotado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.stop()
cv2.destroyAllWindows()
