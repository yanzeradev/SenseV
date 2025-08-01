# utils.py
import torch
import torchvision.transforms as T
from PIL import Image
import cv2

from TransReID.config import cfg
from TransReID.model import make_model
from boxmot.utils.ops import letterbox

class TransReIDWrapper(torch.nn.Module):
    """
    Wrapper para o modelo TransReID para facilitar a integração com BoT-SORT.
    Encapsula o carregamento do modelo, pré-processamento e extração de features.
    """
    def __init__(self, config_file, model_path, device):
        super().__init__()
        self.device = device

        # Carrega a configuração do modelo TransReID
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
        """Processa um lote de imagens já recortadas (crops)."""
        if not crops:
            return torch.empty((0, 384), device='cpu') # Dimensão do embedding

        batch = torch.stack([self.transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))) for crop in crops])
        
        with torch.no_grad():
            features = self.model(batch.to(self.device))
        
        return features.cpu()

    def get_features(self, bboxes, img):
        """
        Recorta as imagens com base nas bboxes e extrai as features.
        Esta é a função que o BoT-SORT espera encontrar.
        """
        if bboxes is None or len(bboxes) == 0:
            return torch.empty(0, 384)

        crops = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            # Garante que as coordenadas sejam válidas
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            
            crop = img[y1:y2, x1:x2]
            
            # Pula crops vazios/inválidos
            if crop.shape[0] > 0 and crop.shape[1] > 0:
                crops.append(crop)

        if not crops:
            return torch.empty(0, 384)

        return self.forward(crops)

def preprocess(frame, img_size=640):
    """Função de pré-processamento usando letterbox."""
    return letterbox(frame, new_shape=img_size, auto=False, scaleFill=False)
