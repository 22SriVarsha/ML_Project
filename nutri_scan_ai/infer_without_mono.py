import os
import cv2
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from model_depth import RgbPredictor

'''
This file is for testing the rgb_depth_calorie_predictor.pth model without the mono depth model.
We will test a random sample from the dish_ids available and print the results. It will plot the RGB and depth images for the test sample.
'''
IMAGE_ROOT = os.path.join("data/nutrition5k_dataset", "imagery", "realsense_overhead")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loading our trained model from current folder
rgb_model = RgbPredictor().to(device)
rgb_model.load_state_dict(torch.load("rgb_depth_calorie_predictor.pth", map_location=device))
rgb_model.eval()

# get the created metadata file from the data folder
df = pd.read_csv("data/metadata_clean.csv")

# test a random sample from the dish_ids available
sample = df.sample(1).iloc[0]
dish_id = sample["dish_id"]
# get the actual calories from the sample
true_calories = sample["total_calories"]
# get the folder path for the dish
dishDetails = os.path.join(IMAGE_ROOT, dish_id)
rgb_path = None
depth_path = None

for fname in os.listdir(dishDetails):
    lname = fname.lower()
    full_path = os.path.join(dishDetails, fname)
    if "rgb" in lname and fname.endswith((".png", ".jpg", ".jpeg")):
        rgb_path = full_path

    if "depth_raw" in lname or ("depth" in lname and "color" not in lname):
        depth_path = full_path
# Preprocessing the rgb image for testing
rgbTransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
rgb_pil = Image.open(rgb_path).convert("RGB")
tensor1 = rgbTransform(rgb_pil).unsqueeze(0).to(device)
depthImage = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
depthImage = depthImage / 10000.0
depthImage = cv2.resize(depthImage, (224, 224))
dMin = np.min(depthImage)
dMax = np.max(depthImage)
if dMax > dMin:
    depthImage = (depthImage - dMin) / (dMax - dMin)
dTensor = torch.tensor(depthImage, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    prd = rgb_model(tensor1, dTensor)

prd_cals = prd.item()
rgb_fig = np.array(rgb_pil)
plt.figure(figsize=(12, 5))

# plotting RGB Image
plt.subplot(1, 2, 1)
plt.imshow(rgb_fig)
plt.title("RGB Image for the dish test")
plt.axis("off")


plt.subplot(1, 2, 2)
plt.imshow(depthImage, cmap="gray")
plt.title("Depth Map for the dish test")
plt.axis("off")

plt.tight_layout()
plt.show()


print("\nTest results without mono depth model:")
print(f"Actual Calories    : {true_calories:.2f}")
print(f"Predicted Calories : {prd_cals:.2f}")
print(f"Absolute Error     : {abs(true_calories - prd_cals):.2f}")