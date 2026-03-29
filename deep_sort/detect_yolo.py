
from torchvision import transforms
import numpy as np
import torch
import cv2

class Detect():
    def __init__(self, frame, frame_id, model_detect, model_reid, device):
        self.frame = frame
        self.frame_id = frame_id
        self.model_reid = model_reid
        self.model_detect = model_detect
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_embedding(self, x):
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            x = self.transform(x)
            x = x.unsqueeze(0).to(self.device)
            x = self.model_reid(x)
            x = x.cpu().squeeze().numpy()

        return x

    def __call__(self):
        results = self.model_detect.predict(self.frame, classes=0, device=self.device)
        out = np.zeros(shape=(len(results[0].boxes.xyxy), 522))
        for bbox in range(len(results[0].boxes.xyxy)):
            top_x = results[0].boxes.xyxy[bbox][0].cpu().int()
            top_y = results[0].boxes.xyxy[bbox][1].cpu().int()
            bottom_x = results[0].boxes.xyxy[bbox][2].cpu().int()
            bottom_y = results[0].boxes.xyxy[bbox][3].cpu().int()
            detection = self.frame[top_y:bottom_y, top_x:bottom_x]
            embeddings = self.get_embedding(detection)
            row = np.array([self.frame_id, -1])
            row = np.append(row, [top_x, top_y, bottom_x - top_x, bottom_y - top_y])
            row = np.append(row, [results[0].boxes.conf[bbox].cpu(), -1, -1, -1])
            row = np.append(row, embeddings)
            out[bbox] = row
        return out
