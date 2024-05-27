from flavor_db_utils import get_all_flavordb_info
from edamam_utils import (
    get_edamam_ingredient_info_from_flavordb,
    edamam_and_flavor_db,
    nutritional_edamam_flavor_db
)
from usda_utils import get_usda_info


def get_dataset_flavor_db():
    get_all_flavordb_info()


def get_dataset_edamam():
    get_edamam_ingredient_info_from_flavordb()
    edamam_and_flavor_db()
    nutritional_edamam_flavor_db()


def get_dataset_usda():
    get_usda_info()


def main():
    get_dataset_flavor_db()
    get_dataset_edamam()
    get_dataset_usda()


if __name__ == "__main__":
    """Execute this file only if data is not in your computer. This will
    download the flavor_db (each instance of it), the usda database (the whole
    usda database), and it will use the edamam api, to get the nutritional
    values of each of the flavor_db ingredients.
    If files are in your computer nothing bad should pass neighter."""
    main()
