import cv2
import numpy as np
from ultralytics import YOLO
from tracker.bot_sort import BoTSORT
import argparse

video_path = 'cam4.mp4'
model_crowdhuman = YOLO('modelo_crowdhuman_v12n_16-06-25_adam_imgz640-batch26_200epochs.pt')

# Inicializa o rastreador BoT-SORT
args = argparse.Namespace(
    track_high_thresh=0.6,
    track_low_thresh=0.1,
    new_track_thresh=0.6,
    track_buffer=30,
    match_thresh=0.8,
    aspect_ratio_thresh=1.6,
    min_box_area=10,
    mot20=False,  # <- Este aqui funciona!
    fast_reid_config=None,
    fast_reid_weights=None,
    fuse_score=False,
    with_reid=False,
    proximity_thresh=0.5,
    appearance_thresh=0.25,
    cmc_method='sparseOptFlow',
    name='botsort_demo',
    ablation='none',
    det_thresh=0.4,
    track_thresh=0.5,
    track_frame_rate=30
)

tracker = BoTSORT(
    args, frame_rate=30.0
)

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    # Detecção com YOLOv8
    results = model_crowdhuman.predict(frame, iou=0.6, conf=0.55, classes=[0, 1], verbose=False)

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append([x1, y1, x2, y2, conf, cls_id])
    
    # Converte para np.array no formato que o BoTSORT espera
    detections = np.array(detections)

    # Atualiza rastreador
    tracks = tracker.update(detections, frame)

    # Desenhar resultados
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.tlbr)  # Coordenadas da bounding box
        track_id = track.track_id       
        cls_id = -1

        if cls_id == 0:
            label = f'Cabeca ID:{track_id}'
            color = (255, 0, 0)
        else:
            label = f'Pessoa ID:{track_id}'
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('BoT-SORT + CrowdHuman', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
