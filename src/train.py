import mlflow
import mlflow.tensorflow
from model import create_model
from prepare_data import prepare_data
import os


def train():
    mlflow.set_experiment("cats-dogs-classification")
    
    with mlflow.start_run():
        # Подготовка данных
        train_generator, validation_generator = prepare_data()

        # Создание модели
        model = create_model()

        # Логируем параметры
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("loss", "binary_crossentropy")
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("epochs", 20)

        # Вычисляем steps_per_epoch и validation_steps
        steps_per_epoch = len(train_generator.filenames) // train_generator.batch_size
        validation_steps = len(validation_generator.filenames) // validation_generator.batch_size

        # Обучение модели
        history = model.fit(
            train_generator.repeat(),
            steps_per_epoch=steps_per_epoch,
            epochs=20,
            validation_data=validation_generator.repeat(),
            validation_steps=validation_steps
        )

        # Логируем метрики
        mlflow.log_metric("train_accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])

        # Создаем директорию для сохранения модели
        os.makedirs("models/latest", exist_ok=True)

        # Сохраняем модель локально
        model_path = "models/latest/model.keras"
        model.save(model_path)

        # Логируем модель в MLflow
        mlflow.log_artifact(model_path, artifact_path="model")


if __name__ == "__main__":
    train()
