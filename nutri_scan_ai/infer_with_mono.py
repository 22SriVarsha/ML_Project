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
This file is for testing the rgb_depth_calorie_predictor.pth model with the mono depth model.
We will test a random sample from the dish_ids available and print the results. It will plot the RGB and depth images for the test sample.
'''
IMAGE_ROOT = os.path.join("data/nutrition5k_dataset", "imagery", "realsense_overhead")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loading our trained model from current folder
rgb_model = RgbPredictor().to(device)
rgb_model.load_state_dict(torch.load("rgb_depth_calorie_predictor.pth", map_location=device))
rgb_model.eval()

# Load MiDaS depth model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
midas_transform = midas_transforms.small_transform

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
for file_name in os.listdir(dishDetails):
    lower_name = file_name.lower()
    if "rgb" in lower_name and file_name.lower().endswith((".png", ".jpg", ".jpeg")):
        rgb_path = os.path.join(dishDetails, file_name)
        break
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
rgb_np = np.array(rgb_pil)
tensor1 = rgbTransform(rgb_pil).unsqueeze(0).to(device)
# Predict monocular depth using MiDaS
input_batch = midas_transform(rgb_np).to(device)

with torch.no_grad():
    midas_prediction = midas(input_batch)
    midas_prediction = torch.nn.functional.interpolate(
        midas_prediction.unsqueeze(1),
        size=(224, 224),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_path = midas_prediction.cpu().numpy()
predicted_depth = midas_prediction.cpu().numpy()
# Normalize predicted depth to 0–1
dMin = depth_path.min()
dMax = depth_path.max()

if dMax > dMin:
    depthImage = (depth_path - dMin) / (dMax - dMin)
dTensor = torch.tensor(
        depth_path,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(device)
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
print("\nTest results with mono depth model:")
print(f"Actual Calories: {true_calories:.2f}")
print(f"Predicted Calories: {prd_cals:.2f}")
print(f"Absolute Error: {abs(true_calories - prd_cals):.2f}")