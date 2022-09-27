import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import requests
from io import BytesIO

class PythonPredictor:
    def  __init__(self, config):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        
        self.data_transforms = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                self.normalize
            ])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = resnet50(pretrained=False).to(device)
        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 11)).to(device)
        model.load_state_dict(torch.load('weights.h5', map_location=torch.device(device)))
        model.eval()

    def predictLabel(self, file):
        image_size = 224
        img = torch.reshape(file , (1, 3, image_size, image_size))
        return self.model(img)

    def predict(self, img):
        allClasses = ['3D Mask', 'A4', 'Face Mask', 'Live', 'Pad', 'PC', 'Phone', 'Photo', 'Poster', 'Region Mask', 'Upper Body Mask']
        image = requests.get(img["url"]).content
        img_pil = Image.open(BytesIO(image))
        img_tensor = self.data_transforms(img_pil)
        img_tensor.unsqueeze_(0)
        out = self.predictLabel(img_tensor)
        _, predicted = torch.max(out.data, 1)
        allClasses.sort()
        labelPred = allClasses[predicted]
        return labelPred
  