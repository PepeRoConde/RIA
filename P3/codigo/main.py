import cv2
from ultralytics import YOLO

from acciones import crear_robot
from vision import detectar_posicion_brazos, ejecutar_accion_robot
from camara import Camera

# Cargar YOLO 
model = YOLO('yolov8n-pose.pt')

# Inicializar robot
acciones = crear_robot()

# Inicializar cámara en hilo
cam = Camera()

print("Presiona 'q' para salir")

while True:
    frame = cam.get_frame()
    if frame is None:
        continue

    # Voltear imagen
    frame = cv2.flip(frame, 1)

    # Reducir resolución para más FPS
    frame_small = cv2.resize(frame, (640, 480))

    # Procesar YOLO
    results = model(frame_small, verbose=False)

    # Copia del frame para dibujar
    annotated_frame = results[0].plot()

    for result in results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()

            for person_keypoints in keypoints:
                posicion = detectar_posicion_brazos(person_keypoints)

                # Ejecutar acción del robot
                ejecutar_accion_robot(posicion, acciones)

                # Mostrar texto
                cv2.putText(annotated_frame, f"Posición: {posicion}",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                            (0, 255, 255), 3)

    cv2.imshow("YOLO - Control Robobo", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
