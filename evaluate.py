import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model(model, test_dir, img_size=(128, 128), batch_size=32):
    test_gen = ImageDataGenerator(rescale=1./255)
    test_data = test_gen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    loss, accuracy = model.evaluate(test_data)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return test_data

def visualize_predictions(model, data, class_names):
    images, labels = next(data)
    predictions = model.predict(images)
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i])
        pred_label = class_names[np.argmax(predictions[i])]
        true_label = class_names[np.argmax(labels[i])]
        color = 'green' if pred_label == true_label else 'red'
        plt.title(pred_label, color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
