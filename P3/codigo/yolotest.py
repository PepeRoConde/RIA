import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def detect_bottle(frame):
    height, width, _ = frame.shape
    results = model(frame)

    for r in results:
        for i, c in enumerate(r.boxes.cls):
            object_name = model.names[int(c)]

            if object_name == 'bottle':
                box = r.boxes.xyxy[i]

                x1 = box[0].item()
                y1 = box[1].item()
                x2 = box[2].item()
                y2 = box[3].item()

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                norm_x = int(round((center_x / width) * 100))
                norm_y = int(round((center_y / height) * 100))

                return [norm_x, norm_y]

    return [-1, -1]


if __name__ == "__main__":
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)

    if not cap0.isOpened():
        print("Error: No se pudo abrir la cámara 0")
    if not cap1.isOpened():
        print("Error: No se pudo abrir la cámara 1")

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 and not ret1:
            break

        if ret0:
            coords0 = detect_bottle(frame0)
            print("Cam0:", coords0)
            cv2.imshow("YOLOv8 - Camara 0", frame0)

        if ret1:
            coords1 = detect_bottle(frame1)
            print("Cam1:", coords1)
            cv2.imshow("YOLOv8 - Camara 1", frame1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()
