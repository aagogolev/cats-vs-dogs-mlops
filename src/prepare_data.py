from tensorflow.keras.preprocessing.image import ImageDataGenerator

def prepare_data():
    # Пути к данным
    train_dir = 'data/cats_and_dogs_filtered/train'
    validation_dir = 'data/cats_and_dogs_filtered/validation'

    # Аугментация для тренировочных данных
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Только нормализация для валидационных данных
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        shuffle=True
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    return train_generator, validation_generator
