import torch
import torchvision
import cv2
import time
from PIL import Image
from torchvision.transforms import transforms as transforms
import matplotlib
import numpy as np
import os
import pickle


class KeypointsRCNN:

    def __init__(self, min_size: int = 500,GPU=True):
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,
                                                                            num_keypoints=17, min_size=min_size)
        self.transform = transforms.Compose([transforms.ToTensor()])
        if GPU:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        self.model.to(self.device).eval()

    def get_prediction(self, img, threshold=0.9):
        image, orig_frame = self.__image_trasform(img)
        with torch.no_grad():
            outputs = self.model(image)
        return outputs
        pass

    def __image_trasform(self, image):
        pil_image = Image.fromarray(image).convert('RGB')
        orig_frame = image
        image = self.transform(pil_image)
        image = image.unsqueeze(0).to(self.device)
        return image, orig_frame


