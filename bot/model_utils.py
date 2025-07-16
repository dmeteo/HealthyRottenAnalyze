import torch
from torchvision.models import efficientnet_b0
import torch.nn as nn
from PIL import Image
import io
from config import id2label, val_transform, MODEL_PATH, DEVICE

def load_model():
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(id2label))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

def predict_image_from_bytes(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    input_tensor = val_transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(1).item()
    return id2label[pred]
