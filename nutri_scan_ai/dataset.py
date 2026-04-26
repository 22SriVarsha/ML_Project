import os
import cv2
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as tfs


class NutritionLoader(Dataset):
    """Load RGB and depth tensors for a dish and return a scalar calorie target"""

    def __init__(self, csvFile, iRoot):
        self.data = pd.read_csv(csvFile)
        self.iRoot = iRoot
        # ImageNet preprocessing for RGB image
        self.rgb_transform = tfs.Compose([
            tfs.Resize((224, 224)),
            tfs.ToTensor(),
            tfs.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def _change_dirs(self, lName, fName, fP, rgbPath, depthPath):
        """Pick RGB and depth paths from a dish folder """
        if "rgb" in lName and fName.endswith((".png", ".jpg", ".jpeg")):
            rgbPath = fP

        if "depth_raw" in lName or ("depth" in lName and "color" not in lName):
            depthPath = fP

        return rgbPath, depthPath

    def find_files(self, dish_id):
        """Return rgp path and depth path for images"""
        dF = os.path.join(self.iRoot, dish_id)
        if not os.path.exists(dF):
            return None, None
        rgbPath = None
        depthPath = None
        # If multiple files match, take the last match
        for nms in os.listdir(dF):
            lnm = nms.lower()
            fpt = os.path.join(dF, nms)
            rgbPath, depthPath = self._change_dirs(
                lnm, nms, fpt, rgbPath, depthPath
            )

        return rgbPath, depthPath

    def __getitem__(self, idx):
        """Load a sampl withRGB tensor, single-channel depth tensor, and calorie target"""
        row = self.data.iloc[idx]
        dId = row["dish_id"]
        rgbPath, depthPath = self.find_files(dId)
        if rgbPath is None or depthPath is None:
            raise FileNotFoundError(f"No RGB image or depth image found for {dId}")
        rgbImg = Image.open(rgbPath).convert("RGB")
        rgbTensor = self.rgb_transform(rgbImg)
        # original bit depth channels maintenance for depth image
        deptImg = cv2.imread(depthPath, cv2.IMREAD_UNCHANGED).astype(np.float32)

        # scaling the raw depth values for fixed values
        deptImg = deptImg / 10000.0

        deptImg = cv2.resize(deptImg, (224, 224))
        # Resize imagedepth so that it can be joined with RGB image of same dimensions for joining
        # min-max normalization for stabilizing the depth scale
        depthMin = np.min(deptImg)
        depthMax = np.max(deptImg)
        if depthMax > depthMin:
            deptImg = (deptImg - depthMin) / (depthMax - depthMin)

        depthTensor = torch.tensor(deptImg, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor([row["total_calories"]], dtype=torch.float32)
        return rgbTensor, depthTensor, target