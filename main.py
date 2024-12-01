from data_processing import load_data
from eda import visualize_images, get_class_names
from model import build_model
from evaluate import evaluate_model, visualize_predictions
from utils import plot_training_history

train_dir = "dataset/seg_train"
test_dir = "dataset/seg_test"

print("Dataset preparation...")
train_data, val_data = load_data(train_dir)
class_names = get_class_names(train_data)

print("Data structure analysis...")
visualize_images(train_data, class_names)

print("Building model...")
model = build_model()

print("Training model...")
history = model.fit(train_data, validation_data=val_data, epochs=10)

plot_training_history(history)

print("Evaluating model...")
test_data = evaluate_model(model, test_dir)

print("Visualizing predictions...")
visualize_predictions(model, test_data, class_names)
