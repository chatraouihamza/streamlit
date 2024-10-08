import kaggle

# Set your Kaggle API credentials (replace 'username' and 'key' with your own credentials)
kaggle.api.authenticate()

# Specify the dataset to download (without the leading slash)
dataset_name = 'uciml/breast-cancer-wisconsin-data'

# Specify the path where you want to save the downloaded files
download_path = './Data'

# Download dataset files and unzip to the specified path
kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)


