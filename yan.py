import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot import BotSort
from boxmot.utils.ops import letterbox
import torch

# --- CONFIGURAÇÕES ---
VIDEO_PATH = 'projeto_cam_ufcat/cam4.mp4'
YOLO_MODEL_PATH = 'modelo_genero_v12m_28-05-25_adam_imgz640-batch10_200epochs.pt'
REID_MODEL_PATH = 'osnet_ain_x1_0_msmt17.pt'

# --- INICIALIZAÇÃO ---
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = YOLO(YOLO_MODEL_PATH).to(device)

# Mapeamento de classes para cores (BGR)
CLASS_COLORS = {
    0: (255, 0, 0),    # Azul para classe 0
    1: (0, 255, 0),    # Verde para classe 1
    2: (0, 165, 255)  # Laranja para classe 2
}

# Nomes das classes (substitua pelos seus nomes reais)
CLASS_NAMES = {
    0: "Homem",
    1: "Mulher",
    2: "NaoIdentificado"
}

tracker = BotSort(
    reid_weights=Path(REID_MODEL_PATH),
    device=device,
    half=False,
    track_buffer=120,
    appearance_thresh=0.25,
    match_thresh=0.8,
    new_track_thresh=0.7,
    cmc_method='ecc'
)

vid = cv2.VideoCapture(VIDEO_PATH)

def preprocess(frame, img_size=640):
    return letterbox(frame, new_shape=img_size, auto=False, scaleFill=False)

# --- PROCESSAMENTO ---
while True:
    ret, frame = vid.read()
    if not ret:
        break

    # Pré-processamento
    frame_processed, ratio, (dw, dh) = preprocess(frame)
    
    # Detecção
    results = model.predict(frame_processed, stream=True, verbose=False, device=device)
    
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()  # Extrai o valor float do tensor
            cls = int(box.cls[0].item())
            
            # Ajuste de coordenadas
            x1 = (x1 - dw) / ratio[0]
            y1 = (y1 - dh) / ratio[1]
            x2 = (x2 - dw) / ratio[0]
            y2 = (y2 - dh) / ratio[1]
            
            detections.append([x1, y1, x2, y2, conf, cls])

    detections = np.array(detections) if detections else np.empty((0, 6))
    
    # Tracking
    tracks = tracker.update(detections, frame)

    # Desenhar resultados
    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        track_id = int(track[4])
        conf = track[5]
        cls = int(track[6])
        
        # Obter cor e nome da classe
        color = CLASS_COLORS.get(cls, (0, 255, 255))  # Amarelo como fallback
        cls_name = CLASS_NAMES.get(cls, f"Classe {cls}")
        
        # Bounding Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Label
        label = f"{cls_name} ID:{track_id} Conf:{conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        
        # Fundo do label
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        # Texto
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    cv2.imshow('Tracking com Classes', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
