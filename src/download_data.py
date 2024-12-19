import os
import zipfile
import requests

def download_dataset():
    # URL датасета с Kaggle (используем упрощенную версию)
    url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    
    # Создаем директорию для данных
    os.makedirs("data", exist_ok=True)
    
    # Загружаем архив
    response = requests.get(url)
    with open("data/cats_and_dogs.zip", "wb") as f:
        f.write(response.content)
    
    # Распаковываем архив
    with zipfile.ZipFile("data/cats_and_dogs.zip", "r") as zip_ref:
        zip_ref.extractall("data")
    
    # Удаляем архив
    os.remove("data/cats_and_dogs.zip")

if __name__ == "__main__":
    download_dataset()
