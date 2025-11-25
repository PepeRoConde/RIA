from stable_baselines3 import SAC
from ultralytics import YOLO

def carga_politica(politica_ruta, entorno):
    modelo = SAC.load(politica_ruta, env=entorno)
    print(f"Modelo cargado de {politica_ruta}")
    return modelo

def carga_modelo():
    return YOLO('yolov8n-pose.pt')
