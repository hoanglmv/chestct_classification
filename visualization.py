import matplotlib.pyplot as plt
from PIL import Image
import random
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
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


def plot_predictions(model,label_encoder,x_sample):
    # Predict labels
    y_pred_probs = model.predict(x_sample)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Decode labels using label_incoder
    actual_labels = label_encoder.inverse_transform(y_true)
    predicted_labels = label_encoder.inverse_transform(y_pred)
    
    # Plotting
    plt.figure(figsize=(15, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        img = cv2.cvtColor((x_sample[i] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)  # if images are normalized
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"True: {actual_labels[i]}\nPred: {predicted_labels[i]}", fontsize=10, color='green' if actual_labels[i]==predicted_labels[i] else 'red')
    plt.tight_layout()
    plt.show()