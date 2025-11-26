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

with ui.start():
    try:
        while True:

            frame_webcam = camara_webcam.get_frame() if camara_webcam is not None else None
            frame_smartphone = camara_smartphone.get_frame() if camara_smartphone is not None else None
            
            if frame_webcam is not None or not config['mundo_real']:
                accion = modelo.predict(frame_webcam, observacion)
                observacion, recompensa, terminated, truncated, info = entorno.step(accion)
    
    except KeyboardInterrupt:
        print("\n=== INTERRUPCIÓN POR USUARIO ===")
    
    except Exception as e:
        # This catches ANY error and prints it
        print("\n=== ERROR NO CONTROLADO ===")
        print(type(e).__name__, "-:-", e)

    finally:
        limpia_recursos(camara_webcam, camara_smartphone)
