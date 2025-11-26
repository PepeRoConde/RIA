import cv2
import yaml
from ui import ui
from camara import CamaraWebcam, CamaraSmartphone
from utils import limpia_recursos
from Entorno import Entorno 
from Modelo import Modelo
import torch

with open("P3/configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)

camara_webcam, camara_smartphone = CamaraWebcam(), CamaraSmartphone()

if camara_webcam is None:
    print("[WARNING] Webcam no disponible - telecontrol deshabilitado")
if camara_smartphone is None:
    print("[WARNING] Smartphone no disponible - detección de objetos deshabilitada")

entorno = Entorno(
    mundo_real=config['mundo_real'],
    camara=camara_smartphone if config['mundo_real'] else None,
    clase_objeto=config.get('clase_objeto', 'cup'),
    visualizar_detecciones=config.get('visualizar_detecciones', False)
)

modelo = Modelo(
    config['ruta_politica'], 
    entorno,
    camara_telecontrol=camara_webcam  
)

observacion, _ = entorno.reset()

print("=== INICIANDO LOOP PRINCIPAL ===")
print("Mostrando ambas cámaras:")
print("- Webcam (izquierda): Para telecontrol con gestos")
print("- Smartphone (derecha): Para detección de objetos")

with ui.start():
    try:
        while True:

            frame_webcam = camara_webcam.get_frame() if camara_webcam is not None else None
            frame_smartphone = camara_smartphone.get_frame() if camara_smartphone is not None else None
            
            if frame_webcam is not None or not config['mundo_real']:
                accion = modelo.predict(frame_webcam, observacion)
                observacion, recompensa, terminated, truncated, info = entorno.step(accion)

                if terminated:
                    observacion, _ = entorno.reset()
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n=== INTERRUPCIÓN POR USUARIO ===")
    
    finally:
        limpia_recursos(camara_webcam, camara_smartphone)
