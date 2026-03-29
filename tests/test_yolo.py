import cv2
import torchreid
import torch.nn as nn
from deep_sort.detect_yolo import Detect
from ultralytics import YOLO


model_detect = YOLO('../weights/yolov8n.pt')
model_reid = torchreid.models.build_model('osnet_ain_x1_0', num_classes=1, pretrained=False)
torchreid.utils.load_pretrained_weights(model_reid, '../weights/osnet_ain_x1_0_msmt17_256x128.pth')
model_reid.classifier = nn.Identity()
model_reid.eval()

image = cv2.imread("../MOT16/train/MOT16-02/img1/000001.jpg", cv2.IMREAD_COLOR)
detect = Detect(image, 1, model_detect, model_reid)
print(detect())