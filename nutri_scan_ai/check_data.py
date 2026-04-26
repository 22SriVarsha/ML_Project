import os

data_folder = "data/nutrition5k_dataset"

folders = [
    "metadata",
    "dish_ids/splits",
    "imagery/realsense_overhead"
]

for folder in folders:
    fpr = os.path.join(data_folder, folder)

mpr = os.path.join(data_folder, "metadata")
print("Metadata files are:", os.listdir(mpr))

spth = os.path.join(data_folder, "dish_ids/splits")
print("Split files are available in the folder:", os.listdir(spth))

image_path = os.path.join(data_folder, "imagery/realsense_overhead")
print("Sample dish files are available in the folder:", os.listdir(image_path)[:5])