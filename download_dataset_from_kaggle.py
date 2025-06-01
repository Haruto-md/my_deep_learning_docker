import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_kaggle_dataset(competition_name, download_path="./input"):
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Ensure the download path exists
    os.makedirs(download_path, exist_ok=True)

    # Download the dataset
    print(f"Downloading dataset from competition: {competition_name}")
    api.competition_download_files(competition_name, path=download_path)
    print(f"Dataset downloaded to: {download_path}")

if __name__ == "__main__":
    # Replace 'your-competition-name' with the actual competition name
    competition_name = "your-competition-name"
    download_kaggle_dataset(competition_name)