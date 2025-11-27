import cv2
import yaml
from ui import ui
from camara import CamaraWebcam, CamaraSmartphone
from utils import limpia_recursos, config
from Entorno import Entorno 
from Modelo import Modelo
import traceback

camara_webcam = CamaraWebcam(1) 
camara_smartphone = None if config['mundo_real'] else CamaraSmartphone()
#camara_smartphone = None

entorno = Entorno(
    ip=config['ip'],
    mundo_real=config['mundo_real'],
    camara=camara_smartphone,
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

            frame_webcam = camara_webcam.get_frame() if camara_webcam else None
            if frame_webcam is not None or not config['mundo_real']:
                accion = modelo.predict(frame_webcam, observacion)
                observacion, recompensa, terminated, truncated, info = entorno.step(accion) 

    except KeyboardInterrupt:
        entorno.desconecta()
        print("\n=== INTERRUPCIÓN POR USUARIO ===")
    
    except Exception as e:
        # This catches ANY error and prints it
        print("\n=== ERROR NO CONTROLADO ===")
        print(type(e).__name__, "->:<-", e)
        traceback.print_exc()

    finally:
        limpia_recursos(camara_webcam, camara_smartphone)
