from pathlib import Path

import pandas as pd

from src.evaluation.image_utils import crop_widget


DATASET_FOLDER = Path("data/raw/xmls")
OUTPUT_FOLDER = Path("results/test_crops")

app = "02_auto_01_carmax"

old_node = 9
new_node = 11


# File paths
old_csv = DATASET_FOLDER / f"{app}_old.csv"
new_csv = DATASET_FOLDER / f"{app}_new.csv"

old_image = DATASET_FOLDER / f"{app}_old.png"
new_image = DATASET_FOLDER / f"{app}_new.png"


# Load CSV files
old_df = pd.read_csv(old_csv)
new_df = pd.read_csv(new_csv)


# Get the Ground Truth widgets
old_widget = old_df.iloc[old_node]
new_widget = new_df.iloc[new_node]


# Read widget bounds
old_bounds = old_widget.get("bounds")
new_bounds = new_widget.get("bounds")


# Crop the widgets from the screenshots
old_crop = crop_widget(
    old_image,
    old_bounds
)

new_crop = crop_widget(
    new_image,
    new_bounds
)


# Create the output folder
OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# Save the test crops
if old_crop is not None:
    old_crop.save(
        OUTPUT_FOLDER / "carmax_make_model_old.png"
    )

if new_crop is not None:
    new_crop.save(
        OUTPUT_FOLDER / "carmax_make_model_new.png"
    )


print("Old text:", old_widget.get("text"))
print("Old bounds:", old_bounds)

print("New text:", new_widget.get("text"))
print("New bounds:", new_bounds)

print("Test crops saved to:", OUTPUT_FOLDER)