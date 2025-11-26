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
            # Get frames from both cameras
            frame_webcam = None
            frame_smartphone = None
            
            if camara_webcam is not None:
                frame_webcam = camara_webcam.get_frame()
            
            if camara_smartphone is not None:
                frame_smartphone = camara_smartphone.get_frame()
            
            # Display both camera feeds
            # if frame_webcam is not None and frame_smartphone is not None:
            if True:
            # Add labels to frames
                frame_webcam_labeled = frame_webcam.copy()
                frame_smartphone_labeled = frame_smartphone.copy()
                
                cv2.putText(frame_webcam_labeled, "WEBCAM - Telecontrol", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame_smartphone_labeled, "SMARTPHONE - Deteccion", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frames side by side
                combined_frame = cv2.hconcat([frame_webcam_labeled, frame_smartphone_labeled])
                cv2.imshow("Dual Camera View", combined_frame)
            
            elif frame_webcam is not None:
                # Only webcam available
                frame_webcam_labeled = frame_webcam.copy()
                cv2.putText(frame_webcam_labeled, "WEBCAM - Telecontrol", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Webcam View", frame_webcam_labeled)
            
            elif frame_smartphone is not None:
                # Only smartphone available
                frame_smartphone_labeled = frame_smartphone.copy()
                cv2.putText(frame_smartphone_labeled, "SMARTPHONE - Deteccion", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Smartphone View", frame_smartphone_labeled)
            
            # Process telecontrol if webcam frame is available
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
