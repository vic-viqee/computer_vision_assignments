import csv
import os
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


class ShapeDataset(Dataset):
    def __init__(self, root, size=128, train=False):
        self.root = root
        self.image_dir = os.path.join(root, "images")
        self.size = size
        self.train = train
        self.rows = []

        labels_path = os.path.join(root, "labels.csv")
        with open(labels_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = os.path.join(self.image_dir, row["filename"])
        img = Image.open(path).convert("RGB")
        img = img.resize((self.size, self.size))

        box = [
            float(row["cx"]),
            float(row["cy"]),
            float(row["w"]),
            float(row["h"]),
        ]
        is_object = 0.0 if row["shape"] == "none" else 1.0

        if self.train:
            img, box = self.augment(img, box)

        # PIL gives uint8 HWC. torch wants float CHW.
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        image = torch.from_numpy(arr).permute(2, 0, 1)

        target = torch.tensor(
            [is_object, box[0], box[1], box[2], box[3]],
            dtype=torch.float32,
        )
        return image, target

    def augment(self, img, box):
        # simple augmentation only. no rotation because the box
        # would have to be rotated too and that gets messy.
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if box[2] > 0:
                box[0] = 1.0 - box[0]

        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            img = ImageEnhance.Brightness(img).enhance(factor)

        if random.random() < 0.3:
            factor = random.uniform(0.8, 1.2)
            img = ImageEnhance.Contrast(img).enhance(factor)

        return img, box
