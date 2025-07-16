import torch
from torchvision.models import efficientnet_b0
import torch.nn as nn


def load_model_effnet():
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 28)
    model.load_state_dict(torch.load('best_vegetables_effnetb0.pth', map_location='cpu'))
    model.eval()
    print('OK, модель загрузилась и готова к инференсу!')

load_model_effnet()