import cv2
import numpy as np
from ultralytics import YOLO

class SensorObjeto:
    """
    Detecta objetos usando una cámara en lugar de los sensores del Robobo.
    Útil para el mundo real donde los sensores de blob pueden no funcionar bien.
    """
    def __init__(self, modelo_yolo='yolov8n.pt', clase_objetivo='cup'):
        """
        Args:
            modelo_yolo: Ruta al modelo YOLO a usar
            clase_objetivo: Nombre de la clase a detectar (ej: 'cup', 'bottle', 'ball')
        """
        self.modelo = YOLO(modelo_yolo, verbose=False)
        self.clase_objetivo = clase_objetivo
        self.frame_width = 640
        self.frame_height = 480
        
    def detectar_objeto(self, frame):
        """
        Detecta el objeto objetivo en el frame.
        
        Args:
            frame: Frame de OpenCV (numpy array)
            
        Returns:
            tuple: (x, y, tamano) donde:
                - x: coordenada x normalizada [0-100] (-1 si no detectado)
                - y: coordenada y normalizada [0-100] (-1 si no detectado)
                - tamano: área del bounding box en pixels (-1 si no detectado)
        """
        if frame is None:
            return -1, -1, -1
            
        # Ejecutar detección
        resultados = self.modelo(frame, verbose=False)
        
        if len(resultados) == 0 or len(resultados[0].boxes) == 0:
            return -1, -1, -1
        
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
            return -1, -1, -1
        
        # Extraer coordenadas del bounding box
        x1, y1, x2, y2 = mejor_deteccion.xyxy[0].cpu().numpy()
        
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
        
        return x_norm, y_norm, tamano
    
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
