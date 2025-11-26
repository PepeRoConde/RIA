import cv2
import threading
import platform
import time
import yaml

with open("P3/configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)

class Camara:
    def __init__(self, src=None, nombre="Camara", target_fps=30):
        """
        Inicializa una cámara con control de frame rate.
        
        Args:
            src: Índice de la cámara (0, 1, 2...) o None para auto-detectar
            nombre: Nombre descriptivo para esta cámara (ej: "Webcam", "Smartphone")
            target_fps: Frame rate objetivo para la captura
        """
        self.nombre = nombre
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.last_capture_time = 0
        self.consecutive_errors = 0
        self.max_consecutive_errors = 10

        if src is None:
            src = 1 if platform.system() == "Darwin" else 0

        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara '{nombre}' con src={src}")

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['frame_x'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['frame_y'])

        # Try to set camera FPS (not all cameras support this)
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)

        self.frame = None
        self.running = True
        self.frame_count = 0
        self.start_time = time.time()

        # Threading
        self.lock = threading.Lock()
        
        # Start background capture thread
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        
        print(f"[Camara] '{nombre}' inicializada correctamente (src={src}, target_fps={target_fps})")

    def update(self):
        """Actualiza el frame en un hilo en segundo plano con control de frame rate"""
        while self.running:
            current_time = time.time()
            
            # Only capture if enough time has passed since last capture
            if current_time - self.last_capture_time >= self.frame_interval:
                ret, frame = self.cap.read()
                
                if ret:
                    self.consecutive_errors = 0
                    with self.lock:
                        self.frame = frame
                    self.frame_count += 1
                    self.last_capture_time = current_time
                else:
                    self.consecutive_errors += 1
                    if self.consecutive_errors >= self.max_consecutive_errors:
                        print(f"Error: {self.nombre} - Demasiados errores consecutivos ({self.consecutive_errors})")
                        # Don't break, keep trying but log the issue
                    
                    # On error, wait a bit before retrying
                    time.sleep(0.1)
            else:
                # Calculate precise sleep time until next frame
                sleep_time = self.frame_interval - (current_time - self.last_capture_time)
                if sleep_time > 0:
                    time.sleep(sleep_time * 0.5)  # Sleep half the time to be responsive

    def get_frame(self):
        """Obtiene el frame actual (con flip horizontal)"""
        with self.lock:
            if self.frame is None:
                return None
            return cv2.flip(self.frame, 1)
    
    def get_frame_raw(self):
        """Obtiene el frame sin procesar (sin flip)"""
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def get_actual_fps(self):
        """Calcula el FPS real basado en los frames capturados"""
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            return self.frame_count / elapsed_time
        return 0

    def get_stats(self):
        """Retorna estadísticas de la cámara"""
        return {
            'nombre': self.nombre,
            'target_fps': self.target_fps,
            'actual_fps': self.get_actual_fps(),
            'frame_count': self.frame_count,
            'consecutive_errors': self.consecutive_errors,
            'running_time': time.time() - self.start_time
        }

    def stop(self):
        """Detiene la captura y libera recursos"""
        print(f"[Camara] Deteniendo '{self.nombre}'...")
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap.isOpened():
            self.cap.release()
        
        stats = self.get_stats()
        print(f"[Camara] '{self.nombre}' detenida")
        print(f"  - Frames totales: {stats['frame_count']}")
        print(f"  - FPS real: {stats['actual_fps']:.2f}")
        print(f"  - Tiempo ejecución: {stats['running_time']:.2f}s")


class CamaraWebcam(Camara):
    def __init__(self, src=None, target_fps=30):
        if src is None:
            src = 1 if platform.system() == "Darwin" else 0
        super().__init__(src=src, nombre="Webcam Telecontrol", target_fps=target_fps)


class CamaraSmartphone(Camara):
    def __init__(self, src=None, target_fps=30):
        if src is None:
            src = 0 if platform.system() == "Darwin" else 1
        super().__init__(src=src, nombre="Smartphone Detección", target_fps=target_fps)


# Example usage and test
if __name__ == "__main__":
    def test_camera():
        try:
            # Test with 15 FPS target
            cam = CamaraWebcam(target_fps=15)
            
            # Run for 5 seconds
            start_time = time.time()
            frames_processed = 0
            
            while time.time() - start_time < 5:
                frame = cam.get_frame()
                if frame is not None:
                    frames_processed += 1
                    # Display frame
                    cv2.imshow('Test Camera', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                time.sleep(0.01)  # Small sleep to prevent busy waiting in main thread
            
            # Print statistics
            stats = cam.get_stats()
            print("\n--- Camera Statistics ---")
            for key, value in stats.items():
                print(f"{key}: {value}")
            
            cam.stop()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error during test: {e}")

    test_camera()
