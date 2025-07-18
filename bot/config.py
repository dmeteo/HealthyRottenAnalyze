import os
from dotenv import load_dotenv
from torchvision import transforms

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")
MODEL_PATH = os.getenv("MODEL_PATH")
DEVICE = os.getenv("DEVICE", "cpu")

id2label = {
    0: 'Apple__Healthy',
    1: 'Apple__Rotten',
    2: 'Banana__Healthy',
    3: 'Banana__Rotten',
    4: 'Bellpepper__Healthy',
    5: 'Bellpepper__Rotten',
    6: 'Carrot__Healthy',
    7: 'Carrot__Rotten',
    8: 'Cucumber__Healthy',
    9: 'Cucumber__Rotten',
    10: 'Grape__Healthy',
    11: 'Grape__Rotten',
    12: 'Guava__Healthy',
    13: 'Guava__Rotten',
    14: 'Jujube__Healthy',
    15: 'Jujube__Rotten',
    16: 'Mango__Healthy',
    17: 'Mango__Rotten',
    18: 'Orange__Healthy',
    19: 'Orange__Rotten',
    20: 'Pomegranate__Healthy',
    21: 'Pomegranate__Rotten',
    22: 'Potato__Healthy',
    23: 'Potato__Rotten',
    24: 'Strawberry__Healthy',
    25: 'Strawberry__Rotten',
    26: 'Tomato__Healthy',
    27: 'Tomato__Rotten'
}


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
