import os
from torchvision.models import resnet50, ResNet50_Weights
import torch

## model load
# pytorch official resnet50 model (example)
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()
## model path set
now_path = os.path.dirname(os.path.abspath(__file__))
TRITON_PATH = os.path.join(now_path, "models")
TRITON_MODEL_PATH = os.path.join(TRITON_PATH, "core", "1", "model.pt")

## model save
# torchscript .pt file
script_model = torch.jit.script(model)
torch.jit.save(script_model, TRITON_MODEL_PATH)

