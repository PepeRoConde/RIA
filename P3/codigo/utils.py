from stable_baselines3 import SAC
from ultralytics import YOLO
import cv2
import torch

def get_device():
    """Determine the best available device"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def carga_politica(politica_ruta, entorno):
    modelo = SAC.load(politica_ruta, env=entorno)
    print(f"Modelo cargado de {politica_ruta}")
    return modelo

def carga_modelo_YOLO(pose=True):
    device = get_device()
    print(f"[YOLO] Usando dispositivo: {device}")
    if pose:
        model = YOLO('yolov8n-pose.pt', verbose=False)
    else:
        model = YOLO('yolov8n.pt', verbose=False)
    model.to(device)
    return model

def esta_viendo(observacion):
   x, y = observacion['blob_xy'][0], observacion['blob_xy'][1] 
   if x == -1 or x == 101 or x == 0: return False
   else:
       cv2.destroyWindow("YOLO - Telecontrol")
       return True

def muestra(frame_anotado, titulo, posicion=None):
    if posicion: 
        cv2.putText(frame_anotado, f"Posicion: {posicion}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
    cv2.imshow(f"YOLO - {titulo}", frame_anotado)
    cv2.waitKey(1)

def limpia_recursos(camara_webcam, camara_smartphone):
    print("=== LIMPIANDO RECURSOS ===")
    if camara_webcam is not None:
        camara_webcam.stop()
    if camara_smartphone is not None:
        camara_smartphone.stop()
    cv2.destroyAllWindows()
    print("Programa finalizado")

def muestra_doble(frame_webcam, frame_smartphone):
    frame_webcam_labeled = frame_webcam.copy()
    frame_smartphone_labeled = frame_smartphone.copy()
    
    cv2.putText(frame_webcam_labeled, "WEBCAM - Telecontrol", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame_smartphone_labeled, "SMARTPHONE - Deteccion", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    combined_frame = cv2.hconcat([frame_webcam_labeled, frame_smartphone_labeled])
    cv2.imshow("Dual Camera View", combined_frame)
    cv2.waitKey(1)
