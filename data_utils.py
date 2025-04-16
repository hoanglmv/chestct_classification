import os
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
data_path = Path(f"Data")
train_path = data_path / 'train'
test_path = data_path / "test"
val_path = data_path / "valid"

print(os.listdir(data_path))
def load_data(path, data):
    """
    Load the data from dataset into a Data frame.

    Args:
        path (str) : Path to data directory.
        data (str) : What type of data it is.
    Returns:
        df (dataframe) : Pandas dataframe with 2 columns, one column for image path and the other column for image label
    """
    if data == "test":
        squamous_cases_dir =  path / "squamous.cell.carcinoma"
        normal_cases_dir = path / "normal"
        large_cases_dir = path / "large.cell.carcinoma"
        adenocarcinoma_cases_dir = path / "adenocarcinoma"
    else:    
        # train / validation data
        squamous_cases_dir =  path / "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa"
        normal_cases_dir = path / "normal"
        large_cases_dir = path / "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa"
        adenocarcinoma_cases_dir = path / "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib"

    # Appending each image full path
    squamous_list = squamous_cases_dir.glob("*.png")
    normal_list = normal_cases_dir.glob("*.png")
    large_list = large_cases_dir.glob("*.png")
    adenocarcinoma_list = adenocarcinoma_cases_dir.glob("*.png")

    squamous_img_list, squamous_label_list = append_labeled_image(squamous_list, "Squamous cell carcinoma")
    normal_img_list, normal_label_list = append_labeled_image(normal_list, "Normal")
    large_img_list, large_label_list = append_labeled_image(large_list, "Large cell carcinoma")
    adenocarcinoma_img_list, adenocarcinoma_label_list = append_labeled_image(adenocarcinoma_list, "Adenocarcinoma")

    nested_data = [squamous_img_list, normal_img_list, large_img_list, adenocarcinoma_img_list]
    nested_labels = [squamous_label_list, normal_label_list, large_label_list, adenocarcinoma_label_list]
    
    data_list = list(itertools.chain.from_iterable(nested_data))
    labels_list = list(itertools.chain.from_iterable(nested_labels))

    # Dataframe of images and labels
    df = pd.DataFrame(data_list)
    df.columns = ["images"]
    df["label"] = labels_list
    df = df.sample(frac = 1).reset_index(drop = True)

    # return dataframe
    return df

def append_labeled_image(_dir, label):
    """
    Add labeled image to 2 distinct lists.

    Args:
        _dir (list) : List of images.
        lbel (str) : image label.

    Returns:
        case_list (list) : List Contains images of specified case.
        label_list : List of specified label for cases. 
    """
    case_list = []
    label_list = []

    for img in _dir: 
        # Append images to case list
        case_list.append(img)
        # Append labels to label list
        label_list.append(label)

    return case_list, label_list

def plot(df):
    sample_images = df[['images', "label"]].sample(10)
    
    plt.figure(figsize = (15, 5))

    for i, (path, label) in enumerate(zip(sample_images['images'], sample_images['label'])):
        img = Image.open(path)
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Label: {label}")
        
    plt.tight_layout()
    plt.show()