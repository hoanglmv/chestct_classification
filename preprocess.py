import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def preprocess_image(image_path, img_size = (224, 224)):
    """
    Reads image path, resize it then normalize it into range [0, 1]

    Args:
        image_path (pathlib.PosixPath) : path of the image need to be converted to string.
        img_size (tuple) : W, H, C.

    Returns:
        img ((n, w, h, c)) : numpy nd array. 
        (n ==> number of training examples),
        (w ==> image width)
        (h ==> image height)
        (c ==> color channel)
    """
    img = cv2.imread(str(image_path))  # Load image (BGR format)
    img = cv2.resize(img, img_size)  # Resize to model input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = img / 255.0   # Normalize to [0, 1]
    return img

def process_dataframe(df, label_encoder, fit_encoder=False):
    """
    Takes pandas dataframe of 2 columns, map each label into integer when fit_encoder is True

    Args:
        df (Dataftrame (2 cols)) : Pandas dataframe.
        label_encoder : LabelEncoder object
        fit_encoder (boolean)

    Returns:
        X (ndarray) : Preprocessed images.
        Y (ndarray) : Integer encoded labels.
    """
    X = np.stack(df["images"].apply(preprocess_image))
    
    if fit_encoder:
        y = label_encoder.fit_transform(df["label"])
    else:
        y = label_encoder.transform(df["label"])
    
    return X, y

# Plot random set of images
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