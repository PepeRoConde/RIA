from stable_baselines3 import SAC
from ultralytics import YOLO
import cv2

def carga_politica(politica_ruta, entorno):
    modelo = SAC.load(politica_ruta, env=entorno)
    print(f"Modelo cargado de {politica_ruta}")
    return modelo

def carga_modelo_YOLO():
    return YOLO('yolov8n-pose.pt', verbose=False)

def esta_viendo(observacion):
   x, y = observacion['blob_xy'][0], observacion['blob_xy'][1] 
   if x == -1 or x == 101 or x == 0: return False
   else: return True

def muestra(frame_anotado, posicion):
    cv2.putText(frame_anotado, f"Posición: {posicion}",
      (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
      (0, 255, 255), 3)
    
    cv2.imshow("YOLO - Control Robobo", frame_anotado)
    cv2.waitKey(1)  # Add this!


