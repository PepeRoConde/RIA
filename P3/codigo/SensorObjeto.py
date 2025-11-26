import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch

class SensorObjeto:
    def __init__(self, modelo_yolo='yolov8n.pt', clase_objetivo='cup'):
        self.clase_objetivo = clase_objetivo
        self.frame_width = 640
        self.frame_height = 480
        
        # Performance optimizations
        self.last_detection_time = 0
        self.cache_duration = 0.2
        self.cached_detection = (-1, -1, -1)
        self.frame_skip = 3
        self.frame_counter = 0
        
        # Device setup
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[SensorObjeto] Usando dispositivo: {self.device}")
        
        # Load model
        self.modelo = YOLO(modelo_yolo, verbose=False)
        self.modelo.to(self.device)
        
    def detectar_objeto(self, frame):
        """
        Detecta el objeto objetivo en el frame (con frame skipping).
        """
        current_time = time.time()
        
        # Skip frames based on config (but always process if no cache exists)
        self.frame_counter += 1
        should_process = (
            self.frame_counter % self.frame_skip == 0 or 
            self.cached_detection is None or
            current_time - self.last_detection_time >= self.cache_duration
        )
        
        if not should_process:
            return self.cached_detection 

        # Use smaller frame for faster processing
        frame_small = cv2.resize(frame, (320, 240))
        
        # Ejecutar detección en MPS
        resultados = self.modelo(frame_small, verbose=False, device=self.device)
        
        if len(resultados) == 0 or len(resultados[0].boxes) == 0:
            self.cached_detection = (-1, -1, -1)
            self.last_detection_time = current_time
            return self.cached_detection
        
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
            self.last_detection_time = current_time
            return self.cached_detection
        
        # Extraer coordenadas del bounding box
        x1, y1, x2, y2 = mejor_deteccion.xyxy[0].cpu().numpy()  # Move to CPU for processing
        
        # Scale coordinates back to original size
        x1 = x1 * (self.frame_width / 320)
        y1 = y1 * (self.frame_height / 240)
        x2 = x2 * (self.frame_width / 320)
        y2 = y2 * (self.frame_height / 240)
        
        # Calcular centro del objeto
        centro_x = (x1 + x2) / 2
        centro_y = (y1 + y2) / 2
        
        # Normalizar a rango [0-100] como hace el Robobo
        x_norm = int((centro_x / self.frame_width) * 100)
        y_norm = int((centro_y / self.frame_height) * 100)
        
        # Calcular tamaño (área del bounding box)
        tamano = int((x2 - x1) * (y2 - y1))
        
        # Asegurar que las coordenadas están en rango válido
        x_norm = np.clip(x_norm, 0, 100)
        y_norm = np.clip(y_norm, 0, 100)
        
        self.cached_detection = (x_norm, y_norm, tamano)
        self.last_detection_time = current_time
        
        return self.cached_detection
     
    def visualizar_deteccion(self, frame, x, y, tamano):
        """
        Dibuja la detección en el frame para debugging.
        
        Args:
            frame: Frame original
            x, y: Coordenadas normalizadas [0-100]
            tamano: Tamaño del objeto
            
        Returns:
            Frame anotado
        """
        frame_viz = frame.copy()
        
        if x != -1 and y != -1:
            # Convertir coordenadas normalizadas a pixels
            px = int((x / 100) * self.frame_width)
            py = int((y / 100) * self.frame_height)
            
            # Dibujar cruz en el centro
            cv2.drawMarker(frame_viz, (px, py), (0, 255, 0), 
                          cv2.MARKER_CROSS, 20, 2)
            
            # Mostrar información
            texto = f"Objeto: ({x}, {y}) Size: {tamano}"
            cv2.putText(frame_viz, texto, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame_viz, "Objeto NO detectado", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame_viz
