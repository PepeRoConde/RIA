import cv2
import threading
import platform
import time

class Camara:
    def __init__(self, src=None):

        if src is None:
            src = 1 if platform.system() == "Darwin" else 0

        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera with src={src}")

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.frame = None
        self.running = True

        # Start background capture thread
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            time.sleep(0.01)  # Reduce CPU usage

    def get_frame(self):
        return self.frame

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)
        if self.cap.isOpened():
            self.cap.release()
