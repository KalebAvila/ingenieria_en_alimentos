import pandas as pd
import requests

from json import JSONDecodeError
from utils import (
    get_general_path,
    join_paths,
    save_as_pickle,
    save_as_json,
    read_json_file,
    read_pickle_with_pandas,
    make_desired_folder,
    check_if_filepath_exists,
)

# Cantidad de entidades en flavorDB, manteniendo el valor arriba del límite.
SEARCH_VALS_MAX = 974

# Path del archivo final de flavoDB
FILE_NAME = "flavor_db.pkl"
DATA_FILE_PATH = "data/raw"
FLAVOR_DB_PATH = "flavor_db"
EXPANDED_FILE_NAME = "flavor_db_expanded.pkl"


def get_json_entity_by_id(id_val):
    """
    Given the url from flavor_db, and an id, we can get the information.
    """
    int_string = id_val
    base_url = 'https://cosylab.iiitd.edu.in/flavordb/entities_json?id='
    url = f'{base_url}{int_string}'
    result = requests.get(url)
    try:
        info = result.json()
    except JSONDecodeError:
        info = {"error": id_val}
    return info


def get_flavor_db_as_df(path):
    """Function to get the flavor_db as a dataFrame"""
    flavor_db = read_pickle_with_pandas(path)
    flavor_db_df = pd.DataFrame(flavor_db).T
    return flavor_db_df


def expand_flavor_db():
    """
    FlavorDB has a lot of molecules, if we want to see this at this granularity
    we need to expand it.
    """
    general_path = get_general_path()
    flavor_db_path = join_paths(general_path, DATA_FILE_PATH, FILE_NAME)
    expanded_flavor_db_path = join_paths(
        general_path, DATA_FILE_PATH, EXPANDED_FILE_NAME
    )
    flavor_db_df = get_flavor_db_as_df(flavor_db_path)
    flavor_db_expanded = []
    for entity in range(flavor_db_df.shape[0]):
        entity_flavor_characteristics_df = pd.DataFrame(
            flavor_db_df.drop(columns='molecules').iloc[entity]).T
        entity_flavor_characteristics_df['entity_index'] = entity
        entity_flavor_df = pd.DataFrame(flavor_db_df.molecules.iloc[entity])
        entity_flavor_df['entity_index'] = entity
        complete_entity_flavor_df = entity_flavor_characteristics_df.merge(
            entity_flavor_df,
            on='entity_index', how='inner'
        ).drop(columns='entity_index')
        flavor_db_expanded.append(complete_entity_flavor_df)
    flavor_db_expanded_whole = pd.concat(flavor_db_expanded)
    save_as_pickle(
        what=flavor_db_expanded_whole,
        where=expanded_flavor_db_path
    )


def get_all_flavordb_info():
    """
    This function will execute the "get_json_entity_by_id" iteratively to
    complete all the flavordb information.
    """
    general_path = get_general_path()
    flavor_db_path = join_paths(general_path, DATA_FILE_PATH, FILE_NAME)
    DATA_FLAVOR_DB_FILE_PATH = join_paths(
        general_path, DATA_FILE_PATH, FLAVOR_DB_PATH
    )
    if not check_if_filepath_exists(DATA_FLAVOR_DB_FILE_PATH):
        make_desired_folder(data_file_path=DATA_FLAVOR_DB_FILE_PATH)

    flavordb = {}
    for index_val in range(SEARCH_VALS_MAX):
        name = f"entity_{index_val}.json"
        entity_json_path = join_paths(DATA_FLAVOR_DB_FILE_PATH, name)
        if not check_if_filepath_exists(entity_json_path):
            flavordb[index_val] = get_json_entity_by_id(index_val)
            save_as_json(what=flavordb[index_val], where=entity_json_path)
        else:
            print(f"File for {name} already exists at {entity_json_path}")
            flavordb[index_val] = read_json_file(entity_json_path)

    # Se itera en cada elemento para verificar si hubo algún error.
    for key in list(flavordb.keys())[:]:
        information = flavordb[key]
        list_of_keys = list(information.keys())
        if 'error' in list_of_keys:
            del flavordb[key]
    # Guardamos el archivo generado como un pickle.
    save_as_pickle(what=flavordb, where=flavor_db_path)


if __name__ == "__main__":
    get_all_flavordb_info()
