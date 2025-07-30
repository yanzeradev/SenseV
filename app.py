import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot import BotSort
from boxmot.utils.ops import letterbox
import torch
import sys

import torch
import torchvision.transforms as T
from PIL import Image

from TransReID.config import cfg
from TransReID.model import make_model

class TransReIDWrapper(torch.nn.Module):
    def __init__(self, config_file, model_path, device):
        super().__init__()
        self.device = device

        # --- LÓGICA DE CARREGAMENTO (JÁ ESTÁ CORRETA) ---
        if config_file:
            cfg.merge_from_file(config_file)
        
        cfg.defrost()
        cfg.MODEL.PRETRAIN_PATH = model_path
        cfg.MODEL.DEVICE = device.type
        cfg.INPUT.SIZE_TEST = [384, 128]
        cfg.MODEL.NAME = 'transformer'
        cfg.MODEL.TRANSFORMER_TYPE = 'vit_base_patch16_224_TransReID'
        cfg.freeze()

        self.model = make_model(cfg, num_class=1, camera_num=1, view_num=1)
        self.model.to(device).eval()

        self.transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, crops):
        """Processa lotes de crops de imagens JÁ RECORTADAS."""
        if not crops:
            return torch.empty((0, 384), device='cpu')

        batch = torch.stack([self.transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))) for crop in crops])
        
        with torch.no_grad():
            features = self.model(batch.to(self.device))
        
        return features.cpu()

    # --- MÉTODO ADICIONADO PARA COMPATIBILIDADE COM BOXMOT ---
    def get_features(self, bboxes, img):
        """
        Recorta as imagens com base nas bboxes e extrai as features.
        Esta é a função que o BoT-SORT espera encontrar.
        """
        if bboxes is None or len(bboxes) == 0:
            return torch.empty(0, 384) # Dimensão do embedding do ViT-Small

        crops = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            # Garante que as coordenadas sejam válidas e dentro dos limites da imagem
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            
            crop = img[y1:y2, x1:x2]
            
            # Pula crops vazios/inválidos que podem ocorrer
            if crop.shape[0] > 0 and crop.shape[1] > 0:
                crops.append(crop)

        if not crops:
            return torch.empty(0, 384)

        return self.forward(crops)

# --- CONFIGURAÇÕES ---
VIDEO_PATH = r'SenseV\cam4.mp4'
YOLO_MODEL_PATH = r'SenseV\modelo_genero_v12m_28-05-25_adam_imgz640-batch10_200epochs.pt'
TRANSREID_MODEL_PATH = r"TransReID\pesos\vit_base.pth" 
TRANSREID_CONFIG_FILE = None
MIN_HITS_FOR_ID = 20

# --- INICIALIZAÇÃO ---
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

model = YOLO(YOLO_MODEL_PATH).to(device)

# Mapeamento de classes para cores (BGR)
CLASS_COLORS = {
    0: (255, 0, 0),    # Azul para classe 0
    1: (0, 255, 0),    # Verde para classe 1
    2: (0, 165, 255) # Laranja para classe 2
}

# Nomes das classes
CLASS_NAMES = {
    0: "Homem",
    1: "Mulher",
    2: "NaoIdentificado"
}

# Inicializa o wrapper do TransReID
transreid = TransReIDWrapper(
    config_file=TRANSREID_CONFIG_FILE,
    model_path=TRANSREID_MODEL_PATH,
    device=device
)

# Inicializa o tracker com o modelo ReID
tracker = BotSort(
    reid_weights=transreid,
    device=device,
    half=True, 
    track_buffer=200,
    appearance_thresh=0.4,
    track_low_thresh = 0.001,
    match_thresh=0.9,
    new_track_thresh=0.85,
    cmc_method='sof',
    fuse_first_associate = True
)


vid = cv2.VideoCapture(VIDEO_PATH)
track_hits = {}

def preprocess(frame, img_size=640):
    return letterbox(frame, new_shape=img_size, auto=False, scaleFill=False)

# --- PROCESSAMENTO ---
while True:
    ret, frame = vid.read()
    if not ret:
        print("Fim do vídeo ou erro na leitura.")
        break

    # Detecção com YOLO
    results = model(frame, verbose=False)
    
    tracks = tracker.update(results[0].boxes.data.cpu().numpy(), frame)

    current_track_ids = set()
    
    if len(tracks) > 0:
        # Desenhar resultados
        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            track_id = int(track[4])
            conf = track[5]
            cls = int(track[6])
            
            # Obter cor e nome da classe
            color = CLASS_COLORS.get(cls, (0, 255, 255)) # Amarelo como fallback
            cls_name = CLASS_NAMES.get(cls, f"Classe {cls}")

            display_id = "?"
            
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
            
    stale_ids = set(track_hits.keys()) - current_track_ids
    for sid in stale_ids:
        del track_hits[sid]

    cv2.imshow('Tracking com Classes', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
