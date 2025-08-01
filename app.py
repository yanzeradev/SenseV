# main.py
import cv2
import torch
from ultralytics import YOLO
from boxmot import BotSort

# Importa os componentes do nosso arquivo de utilidades
from utils import TransReIDWrapper

# --- CONFIGURAÇÕES ---
VIDEO_PATH = r'SenseV\cam4.mp4'
YOLO_MODEL_PATH = r'SenseV\modelo_genero_v12m_28-05-25_adam_imgz640-batch10_200epochs.pt'
TRANSREID_MODEL_PATH = r"TransReID\pesos\vit_base.pth"
TRANSREID_CONFIG_FILE = None # Pode ser definido se necessário

# Mapeamento de classes para cores (BGR) e nomes
CLASS_COLORS = {
    0: (255, 0, 0),      # Azul para classe 0
    1: (0, 255, 0),      # Verde para classe 1
    2: (0, 165, 255)     # Laranja para classe 2
}
CLASS_NAMES = {
    0: "Homem",
    1: "Mulher",
    2: "NaoIdentificado"
}

# --- INICIALIZAÇÃO ---
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# 1. Inicializa o modelo de detecção YOLO
model = YOLO(YOLO_MODEL_PATH).to(device)

# 2. Inicializa o wrapper do TransReID (importado de utils.py)
transreid = TransReIDWrapper(
    config_file=TRANSREID_CONFIG_FILE,
    model_path=TRANSREID_MODEL_PATH,
    device=device
)

# 3. Inicializa o tracker BoT-SORT com o modelo Re-ID
tracker = BotSort(
    reid_weights=transreid,
    device=device,
    half=True,
    track_buffer=200,
    appearance_thresh=0.4,
    track_low_thresh=0.001,
    match_thresh=0.9,
    new_track_thresh=0.85,
    cmc_method='sof',
    fuse_first_associate=True
)

vid = cv2.VideoCapture(VIDEO_PATH)

# --- PROCESSAMENTO ---
while True:
    ret, frame = vid.read()
    if not ret:
        print("Fim do vídeo ou erro na leitura.")
        break

    # Detecção com YOLO
    results = model(frame, verbose=False)
    
    # Atualiza o tracker com as detecções
    # O tracker usará internamente o `transreid.get_features` que definimos
    tracks = tracker.update(results[0].boxes.data.cpu().numpy(), frame)

    if len(tracks) > 0:
        # Desenha os resultados na imagem
        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            track_id = int(track[4])
            conf = track[5]
            cls = int(track[6])
            
            color = CLASS_COLORS.get(cls, (0, 255, 255)) # Amarelo como fallback
            cls_name = CLASS_NAMES.get(cls, f"Classe {cls}")

            # Desenha a bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepara e desenha o label
            label = f"{cls_name} ID:{track_id} Conf:{conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
    cv2.imshow('Tracking com Classes', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- FINALIZAÇÃO ---
vid.release()
cv2.destroyAllWindows()
