import cv2
import threading
import platform
import time

class Camara:
    def __init__(self, src=None, nombre="Camara"):
        """
        Inicializa una cámara.
        
        Args:
            src: Índice de la cámara (0, 1, 2...) o None para auto-detectar
            nombre: Nombre descriptivo para esta cámara (ej: "Webcam", "Smartphone")
        """
        self.nombre = nombre

        if src is None:
            src = 1 if platform.system() == "Darwin" else 0

        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara '{nombre}' con src={src}")

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.frame = None
        self.running = True

        # Start background capture thread
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        self.lock = threading.Lock()
        
        print(f"[Camara] '{nombre}' inicializada correctamente (src={src})")

    def update(self):
        """Actualiza el frame en un hilo en segundo plano"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            time.sleep(0.001)
    
    def get_frame(self):
        """Obtiene el frame actual (con flip horizontal y redimensionado)"""
        with self.lock:
            if self.frame is None:
                return None
            return cv2.resize(cv2.flip(self.frame, 1), (640, 480))
    
    def get_frame_raw(self):
        """Obtiene el frame sin procesar (sin flip ni redimensionado)"""
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        """Detiene la captura y libera recursos"""
        print(f"[Camara] Deteniendo '{self.nombre}'...")
        self.running = False
        self.thread.join(timeout=1)
        if self.cap.isOpened():
            self.cap.release()
        print(f"[Camara] '{self.nombre}' detenida")


class CamaraWebcam(Camara):
    def __init__(self, src=None):
        if src is None:
            src = 1 if platform.system() == "Darwin" else 0
        super().__init__(src=src, nombre="Webcam Telecontrol")


class CamaraSmartphone(Camara):
    def __init__(self, src=None):
        if src is None:
            src = 0 if platform.system() == "Darwin" else 1
        super().__init__(src=src, nombre="Smartphone Detección")
