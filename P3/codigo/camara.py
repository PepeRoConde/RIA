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
            time.sleep(0.01)
    
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
    """Cámara webcam del ordenador - usada para telecontrol"""
    def __init__(self, src=None):
        if src is None:
            # En Mac usa 1 (cámara externa), en otros sistemas usa 0 (cámara integrada)
            src = 1 if platform.system() == "Darwin" else 0
        super().__init__(src=src, nombre="Webcam Telecontrol")


class CamaraSmartphone(Camara):
    """Cámara del smartphone - usada para detección de objetos"""
    def __init__(self, src=None):
        if src is None:
            # En Mac usa 0 (cámara integrada/smartphone), en otros sistemas usa 1
            src = 0 if platform.system() == "Darwin" else 1
        super().__init__(src=src, nombre="Smartphone Detección")


def crear_camaras(config_camaras=None):
    """
    Crea las dos instancias de cámara según la configuración.
    
    Args:
        config_camaras: Diccionario con la configuración:
            {
                'webcam_src': int o None,
                'smartphone_src': int o None
            }
    
    Returns:
        tuple: (camara_webcam, camara_smartphone)
    """
    if config_camaras is None:
        config_camaras = {}
    
    webcam_src = config_camaras.get('webcam_src', None)
    smartphone_src = config_camaras.get('smartphone_src', None)
    
    try:
        camara_webcam = CamaraWebcam(src=webcam_src)
    except RuntimeError as e:
        print(f"[ERROR] No se pudo inicializar la webcam: {e}")
        camara_webcam = None
    
    try:
        camara_smartphone = CamaraSmartphone(src=smartphone_src)
    except RuntimeError as e:
        print(f"[ERROR] No se pudo inicializar la cámara del smartphone: {e}")
        camara_smartphone = None
    
    if camara_webcam is None and camara_smartphone is None:
        raise RuntimeError("No se pudo inicializar ninguna cámara")
    
    return camara_webcam, camara_smartphone


def test_camaras():
    """
    Función de prueba para verificar que ambas cámaras funcionan.
    Muestra ventanas separadas con el feed de cada cámara.
    """
    print("=== TEST DE CÁMARAS ===")
    print("Presiona 'q' para salir")
    
    camaras = {}
    
    # Intentar inicializar ambas cámaras
    try:
        camaras['webcam'] = CamaraWebcam()
    except RuntimeError as e:
        print(f"Webcam no disponible: {e}")
    
    try:
        camaras['smartphone'] = CamaraSmartphone()
    except RuntimeError as e:
        print(f"Smartphone no disponible: {e}")
    
    if not camaras:
        print("No se pudo inicializar ninguna cámara")
        return
    
    print(f"Cámaras activas: {list(camaras.keys())}")
    
    # Esperar un momento para que las cámaras inicialicen
    time.sleep(0.5)
    
    try:
        while True:
            frames_obtenidos = False
            
            for nombre, camara in camaras.items():
                frame = camara.get_frame()
                if frame is not None:
                    # Añadir etiqueta al frame
                    frame_etiquetado = frame.copy()
                    cv2.putText(
                        frame_etiquetado, 
                        f"{camara.nombre}", 
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2
                    )
                    cv2.imshow(f"Test - {camara.nombre}", frame_etiquetado)
                    frames_obtenidos = True
            
            if not frames_obtenidos:
                print("Esperando frames...")
                time.sleep(0.1)
            
            # Salir si se presiona 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Limpiar recursos
        for camara in camaras.values():
            camara.stop()
        cv2.destroyAllWindows()
        print("Test finalizado")


if __name__ == "__main__":
    # Ejecutar test si se corre este archivo directamente
    test_camaras()
