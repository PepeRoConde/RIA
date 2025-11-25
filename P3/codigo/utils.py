from stable_baselines3 import SAC
from ultralytics import YOLO
import cv2

def carga_politica(politica_ruta, entorno):
    modelo = SAC.load(politica_ruta, env=entorno)
    print(f"Modelo cargado de {politica_ruta}")
    return modelo

def carga_modelo_YOLO():
    return YOLO('yolov8n-pose.pt')

def esta_viendo(observacion):
   x, y = observacion['blob_xy'][0], observacion['blob_xy'][1] 
   print(f'x: {x}')
   if x == -1 or x == 101 or x == 0: return False
   else: return True
