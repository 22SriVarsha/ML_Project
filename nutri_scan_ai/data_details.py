import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rootPath = os.path.join("data/nutrition5k_dataset", "imagery", "realsense_overhead")

df = pd.read_csv("data/metadata_clean.csv")

print(df.shape)
print(df.head())
print(df.describe())

#plotting the distributions of the features
features = [
    "total_calories",
    "total_mass",
    "total_fat",
    "total_carb",
    "total_protein"
]

for feature in features:
    plt.figure(figsize=(7, 5))
    plt.hist(df[feature], bins=30)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Number of dishes")
    plt.tight_layout()
    plt.show()

#correlation matrix between the features
correlation_mat = df[features].corr()
plt.figure(figsize=(7, 6))
plt.imshow(correlation_mat)
plt.colorbar()
plt.xticks(range(len(features)), features, rotation=45, ha="right")
plt.yticks(range(len(features)), features)

for i in range(len(features)):
    for j in range(len(features)):
        plt.text(j, i, f"{correlation_mat.iloc[i, j]:.2f}", ha="center", va="center")

plt.title("Correlation Between Nutrition Values")
plt.tight_layout()
plt.show()
#find depth file for each dish
def find_depth_file(dish_id):
    dish_folder = os.path.join(rootPath, dish_id)
    if not os.path.exists(dish_folder):
        return None
    for file_name in os.listdir(dish_folder):
        lower_name = file_name.lower()
        full_path = os.path.join(dish_folder, file_name)
        if "depth_raw" in lower_name or ("depth" in lower_name and "color" not in lower_name):
            return full_path
    return None

depth_rows = []
sample_df = df.head(300)
#calculate depth features for each dish
for _, row in sample_df.iterrows():
    dish_id = row["dish_id"]
    depth_path = find_depth_file(dish_id)
    if depth_path is None:
        continue
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        continue
    depth = depth.astype(np.float32) / 10000.0
    valid_depth = depth[depth > 0]
    if len(valid_depth) == 0:
        continue
    depth_rows.append({
        "dish_id": dish_id,
        "mean_depth": np.mean(valid_depth),
        "min_depth": np.min(valid_depth),
        "max_depth": np.max(valid_depth),
        "depth_range": np.max(valid_depth) - np.min(valid_depth),
        "relative_volume_score": np.sum(valid_depth),
        "total_calories": row["total_calories"],
    })

depth_df = pd.DataFrame(depth_rows)
#write depth features to a csv file
depth_df.to_csv("data/depth_features.csv", index=False)