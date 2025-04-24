#
#   Satvika Eda, Divya Sri Bandaru & Dhriti Anjaria
#   12th April 2025
#   This code is part of the MultiCueStrokeDataset class, which is designed for stroke based colorization.
#

import numpy as np
import cv2
from torchvision import models
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import torch
import random

# This class is used to create a dataset for stroke-based colorization using multiple cues.
class MultiCueStrokeDataset(Dataset):
    def __init__(self, image_paths, img_size=256,
                 strokes_per_object=30, stroke_thickness=(8, 16), device=None):
        self.image_paths = image_paths
        self.img_size = img_size
        self.strokes_per_object = strokes_per_object
        self.stroke_thickness = stroke_thickness
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.resize = T.Resize((img_size, img_size))
        self.to_tensor = T.ToTensor()
        self.to_gray = T.Grayscale(num_output_channels=1)

        # Loading the segmentation model
        self.seg_model = models.segmentation.deeplabv3_resnet50(pretrained=True).to(self.device).eval()
        self.seg_preprocess = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    # Function to draw strokes on the image
    def draw_intent_strokes(self, mask, img_np, stroke_img, count):
        coords = np.argwhere(mask)
        H, W = mask.shape
        if len(coords) == 0:
            return stroke_img

        placed = 0
        attempts = 0
        while placed < count and attempts < count * 5:
            y0, x0 = coords[random.randint(0, len(coords) - 1)]
            color = img_np[y0, x0]

            if not mask[y0, x0]:
                attempts += 1
                continue

            num_points = random.randint(3, 6)
            pts = [(x0, y0)]
            for _ in range(num_points - 1):
                dx = random.randint(-40, 40)
                dy = random.randint(-40, 40)
                x1 = np.clip(pts[-1][0] + dx, 0, W - 1)
                y1 = np.clip(pts[-1][1] + dy, 0, H - 1)
                if not mask[int(y1), int(x1)]:
                    continue
                pts.append((int(x1), int(y1)))

            if len(pts) >= 2:
                thickness = random.randint(*self.stroke_thickness)
                for i in range(len(pts) - 1):
                    cv2.line(stroke_img, pts[i], pts[i + 1], color=tuple(map(int, color)), thickness=thickness)
                placed += 1

            attempts += 1

        return stroke_img

    # Function to get the item from the dataset
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.resize(img)
        img_np = np.array(img)

        with torch.no_grad():
            seg_tensor = self.seg_preprocess(img).unsqueeze(0).to(self.device)
            seg_output = self.seg_model(seg_tensor)['out'][0]
            seg_map = seg_output.argmax(0).cpu().numpy()

        stroke_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        for seg_id in np.unique(seg_map):
            mask = seg_map == seg_id
            if np.sum(mask) < 20:
                continue

            stroke_img = self.draw_intent_strokes(mask, img_np, stroke_img, count=self.strokes_per_object)

        # Convert to LAB
        L_tensor = self.to_tensor(self.to_gray(img))
        lab_hint = cv2.cvtColor(stroke_img, cv2.COLOR_RGB2LAB)
        ab_hint = (lab_hint[:, :, 1:].astype(np.float32) - 128) / 128.0
        ab_hint_tensor = torch.from_numpy(ab_hint.transpose(2, 0, 1)).float()

        lab_gt = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        ab_gt = (lab_gt[:, :, 1:] - 128) / 128.0
        ab_gt_tensor = torch.from_numpy(ab_gt.transpose(2, 0, 1)).float()

        input_tensor = torch.cat([L_tensor, ab_hint_tensor], dim=0)
        return input_tensor, ab_gt_tensor
