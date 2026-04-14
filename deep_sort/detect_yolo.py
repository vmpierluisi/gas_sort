
from torchvision import transforms
import numpy as np
import torch
import cv2
import pickle
import os

class Detect():
    def __init__(self, frame, frame_id, seq_name, model_detect, model_reid, device):
        self.seq_name = seq_name
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

    def get_embeddings_batch(self, crops):
        tensors = []
        for crop in crops:
            x = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(x))

        batch = torch.stack(tensors).to(self.device)  # (N, 3, 256, 128)
        with torch.no_grad():
            embeddings = self.model_reid(batch)  # single forward pass
        return embeddings.cpu().numpy()  # (N, 512)

    def __call__(self):
        cache_path = f"cache/{self.seq_name}_{self.frame_id:06d}.pkl"

        if os.path.exists(cache_path):
            return pickle.load(open(cache_path, "rb"))

        results = self.model_detect.predict(self.frame, classes=0, device=self.device)
        boxes = results[0].boxes
        n = len(boxes.xyxy)
        if n == 0:
            return np.zeros(shape=(0, 522))

        crops = []
        coords = []
        for bbox in range(n):
            top_x = int(boxes.xyxy[bbox][0].cpu())
            top_y = int(boxes.xyxy[bbox][1].cpu())
            bottom_x = int(boxes.xyxy[bbox][2].cpu())
            bottom_y = int(boxes.xyxy[bbox][3].cpu())
            crops.append(self.frame[top_y:bottom_y, top_x:bottom_x])
            coords.append((top_x, top_y, bottom_x, bottom_y))

        embeddings = self.get_embeddings_batch(crops)  # one forward pass

        out = np.zeros(shape=(n, 522))

        for i, (top_x, top_y, bottom_x, bottom_y) in enumerate(coords):
            row = np.array([self.frame_id, -1,
                            top_x, top_y, bottom_x - top_x, bottom_y - top_y,
                            float(boxes.conf[i].cpu()), -1, -1, -1])
            out[i] = np.append(row, embeddings[i])

        os.makedirs("cache", exist_ok=True)
        pickle.dump(out, open(cache_path, "wb"))

        return out