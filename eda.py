import matplotlib.pyplot as plt
import numpy as np

def visualize_images(data, class_names):
    images, labels = next(data)
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i])
        plt.title(class_names[np.argmax(labels[i])])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def get_class_names(data):
    return list(data.class_indices.keys())
