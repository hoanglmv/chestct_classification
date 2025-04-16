import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder

def preprocess_image(image_path, img_size=(224, 224)):
    img = cv2.imread(str(image_path))
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def process_dataframe(df, label_encoder: LabelEncoder, fit_encoder=False):
    """
    Trả về:
      X: np.ndarray shape (n, h, w, c)
      y: np.ndarray shape (n,)
    """
    X = np.stack(df["images"].apply(preprocess_image))
    if fit_encoder:
        y = label_encoder.fit_transform(df["label"])
    else:
        y = label_encoder.transform(df["label"])
    return X, y
