import os
import csv
import pandas as pd

root_path = "data/nutrition5k_dataset"
mFile = os.path.join(root_path, "metadata", "dish_metadata_cafe1.csv")
iRoot = os.path.join(root_path, "imagery", "realsense_overhead")

items = []

def has_rgb_and_depth(d_id):
    dF = os.path.join(iRoot, d_id)
    if not os.path.exists(dF):
        print(dF)
        return False
    eF = os.listdir(dF)
    valid_rggb = any(
        "rgb" in file.lower() and file.lower().endswith((".png", ".jpg", ".jpeg"))
        for file in eF
    )
    # for i in range(len(eF)):
    #     print(eF[i])
    # print(valid_rggb)
    valid_depth = any(
        "depth_raw" in file.lower() or ("depth" in file.lower() and "color" not in file.lower())
        for file in eF
    )
    return valid_rggb and valid_depth

with open(mFile, "r", encoding="utf-8") as file:
    fR = csv.reader(file)
    for row in fR:
        dish_id = row[0]
        if not has_rgb_and_depth(dish_id):
            continue
        items.append({
            "dish_id": dish_id,
            "total_calories": float(row[1]),
            "total_mass": float(row[2]),
            "total_fat": float(row[3]),
            "total_carb": float(row[4]),
            "total_protein": float(row[5])
        })
df = pd.DataFrame(items)
# print("Total dishes:", len(df))
# print(df.head())
os.makedirs("data", exist_ok=True)
df.to_csv("data/metadata_clean.csv", index=False)
print("File created after cleaning the metadata:metadata_clean.csv")