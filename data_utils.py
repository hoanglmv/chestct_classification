import kagglehub
from pathlib import Path
import itertools
import pandas as pd

def download_dataset(dataset_name="mohamedhanyyy/chest-ctscan-images"):
    path = kagglehub.dataset_download(dataset_name)
    print("Path to dataset files:", path)
    return Path(f"{path}/Data")

def append_labeled_image(_dir, label):
    case_list, label_list = [], []
    for img in _dir:
        case_list.append(img)
        label_list.append(label)
    return case_list, label_list

def load_data(path: Path, data_type: str) -> pd.DataFrame:
    """
    Trả về DataFrame với 2 cột: 'images' (Path đến ảnh) và 'label' (chuỗi nhãn)
    """
    if data_type == "test":
        mapping = {
            "Squamous cell carcinoma": "squamous.cell.carcinoma",
            "Normal":                   "normal",
            "Large cell carcinoma":     "large.cell.carcinoma",
            "Adenocarcinoma":           "adenocarcinoma"
        }
    else:
        mapping = {
            "Squamous cell carcinoma": "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa",
            "Normal":                   "normal",
            "Large cell carcinoma":     "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa",
            "Adenocarcinoma":           "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib"
        }

    all_imgs, all_labels = [], []
    for label, folder in mapping.items():
        dir_path = path / folder
        imgs, labels = append_labeled_image(dir_path.glob("*.png"), label)
        all_imgs.append(imgs)
        all_labels.append(labels)

    images = list(itertools.chain.from_iterable(all_imgs))
    labels = list(itertools.chain.from_iterable(all_labels))
    df = pd.DataFrame({"images": images, "label": labels})
    return df.sample(frac=1).reset_index(drop=True)
