import cv2
import numpy as np
from ultralytics import YOLO
import torch

from utils import carga_modelo_YOLO, config



class SensorObjeto:
    def __init__(self, modelo_yolo='yolov8n.pt', clase_objetivo='cup'):
        self.clase_objetivo = clase_objetivo
        self.frame_width = config['frame_x']
        self.frame_height = config['frame_y'] 
        self.factor_tamano = config['factor_tamano'] 
        
        # Performance optimizations
        self.cached_detection = (-1, -1, -1)
        self.frame_skip = 3
        self.frame_counter = 0
        
        # Device setup
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[SensorObjeto] Usando dispositivo: {self.device}")
        
        self.modelo = carga_modelo_YOLO(pose=False)

    def detectar_objeto(self, frame):
        self.frame_counter += 1
        
        if not self.frame_counter % self.frame_skip == 0 or self.cached_detection is None:
            return self.cached_detection 
        
        resultados = self.modelo(frame, verbose=False, device=self.device)
        
        if len(resultados) == 0 or len(resultados[0].boxes) == 0:
            print('YOLO no vio nada. solucionado usando cache')
            if self.cached_detection:
                return self.cached_detection
            else:
                return (-1, -1, -1) 
        
        # Buscar el objeto objetivo con mayor confianza
        mejor_deteccion = None
        mejor_confianza = 0
        
        for box in resultados[0].boxes:
            clase_id = int(box.cls[0])
            clase_nombre = self.modelo.names[clase_id]
            confianza = float(box.conf[0])
            
            if clase_nombre == self.clase_objetivo and confianza > mejor_confianza:
                mejor_confianza = confianza
                mejor_deteccion = box
        
        if mejor_deteccion is None:
            self.cached_detection = (-1, -1, -1)
            return self.cached_detection
        
        # Extraer coordenadas del bounding box
        x1, y1, x2, y2 = mejor_deteccion.xyxy[0].cpu().numpy()  # Move to CPU for processing
        
        # Centros
        centro_x, centro_y  = (x1 + x2) / 2, (y1 + y2) / 2
        
        
        # Normalizar y clipear  
        x_norm = np.clip(int((centro_x / self.frame_width) * 100), 0, 100)
        y_norm = np.clip(int((centro_y / self.frame_height) * 100), 0, 100)

        # Calcular tamaño (área del bounding box)
        tamano = int((x2 - x1) * (y2 - y1))

        print(f'tmano: {tamano}, correjido: {self.factor_tamano * tamano}')
        
        tamano = tamano * self.factor_tamano

        self.cached_detection = (x_norm, y_norm, tamano)
        
        return self.cached_detection
     
    def visualizar_deteccion(self, frame, x, y, tamano):
        frame_viz = frame.copy()
        
        if x != -1 and y != -1:
            # Convertir coordenadas normalizadas a pixels
            px = int((x / 100) * self.frame_width)
            py = int((y / 100) * self.frame_height)
            
            # Dibujar cruz en el centro
            cv2.drawMarker(frame_viz, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            
            # Mostrar información
            texto = f"Objeto: ({x}, {y}) Size: {tamano}"
            cv2.putText(frame_viz, texto, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame_viz, "Objeto NO detectado", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame_viz
