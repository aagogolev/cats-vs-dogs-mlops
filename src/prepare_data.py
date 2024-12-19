import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def prepare_data(data_dir='data/cats_and_dogs_filtered', img_height=150, img_width=150):
    train_dir = f'{data_dir}/train'
    validation_dir = f'{data_dir}/validation'
    
    # Создаем генератор данных с аугментацией для тренировочного набора
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Генератор данных для валидационного набора (только нормализация)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='binary'
    )

    validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='binary'
    )
    
    return train_generator, validation_generator

if __name__ == "__main__":
    prepare_data()
