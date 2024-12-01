import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(train_dir, img_size=(128, 128), batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_data = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        subset='training',
        class_mode='categorical' 
    )

    val_data = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        subset='validation',
        class_mode='categorical'  
    )
    
    return train_data, val_data
