import matplotlib.pyplot as plt
from PIL import Image
import random
import cv2
import numpy as np

def plot_sample_images(df, n=10):
    sample = df.sample(n)[["images", "label"]]
    plt.figure(figsize=(15, 5))
    for i, (path, label) in enumerate(zip(sample['images'], sample['label'])):
        img = Image.open(path)
        plt.subplot(2, n//2, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(label)
    plt.tight_layout()
    plt.show()

def plot_predictions(model, X, y_true, label_encoder, n=10):
    idx = random.sample(range(len(X)), n)
    Xs = X[idx]; ys = y_true[idx]
    probs = model.predict(Xs)
    y_pred = np.argmax(probs, axis=1)
    actual = label_encoder.inverse_transform(ys)
    pred   = label_encoder.inverse_transform(y_pred)

    plt.figure(figsize=(15, 6))
    for i in range(n):
        plt.subplot(2, n//2, i+1)
        img = (Xs[i] * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img); plt.axis('off')
        color = 'green' if actual[i]==pred[i] else 'red'
        plt.title(f"T:{actual[i]}\nP:{pred[i]}", color=color, fontsize=10)
    plt.tight_layout()
    plt.show()
