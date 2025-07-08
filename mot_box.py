import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot import BotSort

# --- CONFIGURAÇÕES ---
VIDEO_PATH = 'projeto_cam_ufcat/cam4.mp4'
YOLO_MODEL_PATH = 'yolov8n.pt'
REID_MODEL_PATH = 'osnet_x0_25_msmt17.pt'

# --- INICIALIZAÇÃO DOS MODELOS ---
model = YOLO(YOLO_MODEL_PATH)
tracker = BotSort(
    reid_weights=Path(REID_MODEL_PATH),
    device='cuda:0',
    half=False # 'half=False' é o mesmo que fp16=False, está correto
)
vid = cv2.VideoCapture(VIDEO_PATH)

# --- PROCESSAMENTO DO VÍDEO ---
while True:
    ret, frame = vid.read()
    if not ret:
        print("Fim do vídeo ou erro na leitura.")
        break

    results = model.predict(frame, stream=True, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = box.cls[0]
            detections.append([float(x1), float(y1), float(x2), float(y2), float(conf), int(cls)])

    detections = np.array(detections)

    # Agora o rastreador receberá o formato correto
    tracker.update(detections, frame)

    tracker.plot_results(frame, show_trajectories=True)

    cv2.imshow('YOLOv8 + BoTSORT', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
