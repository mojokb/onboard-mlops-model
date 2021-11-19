import os
import torch
from net import Net
from bento_service import PytorchModelService

BENTO_PACK_PATH = '/workdir/bentoml'
PATH = './model/model.pt'

model = torch.load(PATH)
model.eval()

svc = PytorchModelService()
svc.pack('model', model)
os.makedirs(BENTO_PACK_PATH, exist_ok=True)
saved_path = svc.save_to_dir(BENTO_PACK_PATH)
