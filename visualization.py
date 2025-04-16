import matplotlib.pyplot as plt
from PIL import Image
import random
import cv2
import numpy as np

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