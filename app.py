import cv2
from ultralytics import YOLO
from tracker import Tracker

video_path = r'cam4.mp4'
output_path = r'output2.mp4'
model_path = r'modelo_genero_v12s_12-03-25_adam_imgz640-batch24_300epochs.pt'

cap = cv2.VideoCapture(video_path)
model = YOLO(model_path)
tracker = Tracker()
ok, frame = cap.read()

cap_out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    cap.get(cv2.CAP_PROP_FPS),
    (frame.shape[1], frame.shape[0])
)

# Mapeamento de track_id para class_id
track_classes = {}

while ok:
    results = model.predict(frame, conf=0.65, iou=0.45)

    for result in results:
        detections = []
        classes_detectadas = []
        for box in result.boxes.data.tolist():
            x0, y0, x1, y1, conf, id_class = box
            detections.append([x0, y0, x1, y1, conf])
            classes_detectadas.append(int(id_class))

        tracker.update(frame, detections)

        # Associa cada track à classe correspondente (na mesma ordem)
        for i, track in enumerate(tracker.tracks):
            if i < len(classes_detectadas):
                track_classes[track.track_id] = classes_detectadas[i]

        for track in tracker.tracks:
            bbox = track.bbox
            x0, y0, x1, y1 = map(int, bbox)
            track_id = int(track.track_id)

            # Recupera class_id do dicionário
            class_idx = track_classes.get(track_id, -1)

            if class_idx == 0:
                class_id = 'Homem'
                class_color = (255, 0, 0) #azul
            elif class_idx == 1:
                class_id = 'Mulher'
                class_color = (0, 0, 255) #vermelho
            else:
                class_id = 'Nao Identificado'
                class_color = (0, 255, 0) #verde

            
            cv2.putText(frame, f'ID: {track_id} Class: {class_id}', (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (x0, y0), (x1, y1), class_color, 2)
            
            cv2.imshow('Tracking', frame)

    cap_out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ok, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()
