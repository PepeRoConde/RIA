import cv2
import yaml
from ui import ui
from ModeloTelecontrol import carga_modelo_telecontrol
from camara import CamaraWebcam, CamaraSmartphone, crear_camaras
from utils import carga_politica
from Entorno import Entorno 
from Modelo import Modelo

with open("P3/configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)

config_camaras = config.get('camaras', {})
camara_webcam, camara_smartphone = crear_camaras(config_camaras)

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
with ui.start():
    try:
        while True:
            frame_webcam = None
            if camara_webcam is not None:
                frame_webcam = camara_webcam.get_frame()
            
            if frame_webcam is None and not config['mundo_real']:
                continue
            
            accion = modelo.predict(frame_webcam, observacion)
            observacion, recompensa, terminated, truncated, info = entorno.step(accion)

            if terminated:
                observacion, _ = entorno.reset()
    
    except KeyboardInterrupt:
        print("\n=== INTERRUPCIÓN POR USUARIO ===")
    
    finally:
        print("=== LIMPIANDO RECURSOS ===")
        if camara_webcam is not None:
            camara_webcam.stop()
        if camara_smartphone is not None:
            camara_smartphone.stop()
        cv2.destroyAllWindows()
        print("Programa finalizado")
