import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import NutritionLoader
from model_depth import RgbPredictor


root_path = os.path.join("data/nutrition5k_dataset", "imagery/realsense_overhead")

cpuorgpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Working on:", cpuorgpu)

nutri_data = NutritionLoader("data/metadata_clean.csv", root_path)
batch = min(1000, len(nutri_data))
nutri_data, _ = random_split(nutri_data, [batch, len(nutri_data) - batch])
# using 80% of the data for training and 20% for testing
train_data_size = int(0.8 * len(nutri_data))
test_data_size = len(nutri_data) - train_data_size
# using random_split to split the data into training and testing sets
train_dataset, test_dataset = random_split(nutri_data, [train_data_size, test_data_size])

load_train = DataLoader(train_dataset, batch_size=8, shuffle=True)
load_test = DataLoader(test_dataset, batch_size=8, shuffle=False)

curr_model = RgbPredictor().to(cpuorgpu)
loss_function = nn.L1Loss()
# using Adam optimizer for training
optmzr = torch.optim.Adam(curr_model.parameters(), lr=0.0001)
epoch_size = 5
for epoch in range(epoch_size):
    curr_model.train()
    total_train_loss = 0

    for rgb, depth, target in load_train:
        rgb = rgb.to(cpuorgpu)
        depth = depth.to(cpuorgpu)
        target = target.to(cpuorgpu)
        optmzr.zero_grad()
        prediction = curr_model(rgb, depth)
        loss = loss_function(prediction, target)
        loss.backward()
        optmzr.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(load_train)

    curr_model.eval()
    total_test_loss = 0

    with torch.no_grad():
        for rgb, depth, target in load_test:
            rgb = rgb.to(cpuorgpu)
            depth = depth.to(cpuorgpu)
            target = target.to(cpuorgpu)
            prediction = curr_model(rgb, depth)
            loss = loss_function(prediction, target)
            total_test_loss += loss.item()
    avg_test_loss = total_test_loss / len(load_test)

    print(
        f"Epoch [{epoch + 1}/{epoch_size}] "
        f"Train MAE: {avg_train_loss:.2f} | "
        f"Test MAE: {avg_test_loss:.2f}"
    )

torch.save(curr_model.state_dict(), "rgb_depth_calorie_predictor.pth")
print("Saved model as rgb_depth_calorie_predictor.pth")