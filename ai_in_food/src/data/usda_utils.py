import os
from urllib.request import urlretrieve
from zipfile import ZipFile


from utils import (
    get_general_path,
    join_paths,
    make_desired_folder,
    check_if_filepath_exists
)

# Path del archivo final de flavoDB
FILE_NAME = "usda.zip"
RAW_DATA_PATH = "data/raw"
DATA_FILE_PATH = "data/raw/usda"
FOCUS_FILE = "branded_food.csv"


def download_link():
    """Download corresponding USDA information"""
    general_path = get_general_path()
    file_path = join_paths(general_path, DATA_FILE_PATH, FILE_NAME)
    main_download_link = "https://fdc.nal.usda.gov/fdc-datasets/" \
                         "FoodData_Central_csv_2023-10-26.zip"
    if not check_if_filepath_exists(file_path):
        urlretrieve(main_download_link, file_path)
    else:
        print(f'File {FILE_NAME} already exists at {file_path}')
    return None


def unzip_file():
    """UNZIP the USDA information previously downloaded"""
    general_path = get_general_path()
    file_path = join_paths(general_path, DATA_FILE_PATH)
    file_pathname = join_paths(general_path, DATA_FILE_PATH, FILE_NAME)
    with ZipFile(file_pathname, "r") as zip_ref:
        zip_ref.extractall(file_path)
    return None


def focus_in_dataset():
    """Get the focus dataset into the consumable level."""
    general_path = get_general_path()
    file_path = join_paths(general_path, DATA_FILE_PATH)
    list_dir = os.listdir(file_path)
    list_dir.remove(FILE_NAME)
    interest_file_path = list_dir[0]
    focus_file_path = join_paths(
        general_path, DATA_FILE_PATH, interest_file_path, FOCUS_FILE
    )
    new_focus_file_path = join_paths(general_path, RAW_DATA_PATH, FOCUS_FILE)
    os.rename(focus_file_path, new_focus_file_path)
    return None


def get_usda_info():
    """Pipeline to get the USDA information."""
    general_path = get_general_path()
    path = join_paths(general_path, DATA_FILE_PATH)
    if not check_if_filepath_exists(path):
        make_desired_folder(data_file_path=path)
    download_link()
    unzip_file()
    focus_in_dataset()


if __name__ == "__main__":
    get_usda_info()
