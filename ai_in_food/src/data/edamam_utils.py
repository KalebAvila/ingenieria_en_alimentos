import json
import pandas as pd
import os
import requests
import time

from difflib import SequenceMatcher

from edamam_variables import found_in_edamam_wierd, not_found_in_edamam
from utils import (
    get_general_path,
    join_paths,
    save_as_pickle,
    save_as_json,
    check_if_filepath_exists,
    read_pickle_with_pandas,
    load_api_keys,
    make_desired_folder,
)

from flavor_db_utils import get_flavor_db_as_df

# Path del archivo final de flavorDB
USEFULL_COLUMNS = [
    'category',
    'entity_id',
    'entity_alias_readable',
    'molecules',
    'search_names'
]

DATA_FILE_PATH = "data/raw"
FLAVORDB_FILE_NAME = "flavor_db.pkl"

DATA_EDAMAM_PATH = "data/raw/edamam/"
DATA_EDAMAM_INGREDIENT_PATH = "data/raw/edamam/ingredients"
DATA_EDAMAM_NUTRIENTS_PATH = "data/raw/edamam/nutrients"


EXPANDED_FILE_NAME = "flavor_db_expanded.pkl"
EDAMAM_FLAVORDB_FILE_NAME = "edamam_flavor_dbs.pkl"
EDAMAM_FLAVORDB_NUTRIENTS_FILE_NAME = "edamam_flavor_dbs_nutrients.pkl"


def get_ingredient_info(ingredient):
    """Get the information of a specific ingredient from edamam api"""
    load_api_keys()
    ingredient = ingredient.lower().replace(' ', '%20').strip()
    API_ID = os.environ["API_ID"]
    API_KEY = os.environ["API_KEY"]
    base_url = 'https://api.edamam.com/api/food-database/v2/parser?'
    url = f'{base_url}app_id={API_ID}&app_key={API_KEY}&' \
          f'ingr={ingredient}&nutrition-type=cooking'
    info = requests.get(url)
    information = info.json()
    return information


def get_ingredient_path(ingredient):
    """Generate the path of the ingredient."""
    general_path = get_general_path()
    path = join_paths(general_path, DATA_EDAMAM_INGREDIENT_PATH)
    make_desired_folder(path)
    ingredient = ingredient.lower().replace(' ', '-')
    filename = f'edamam_{ingredient}.json'
    filepath = join_paths(path, filename)
    return filepath


def process_ingredient_data(ingredient):
    """Given an ingredient, get the path, check if it already exists, and
    finally obtain and save the information if path doesn't exists."""
    filepath = get_ingredient_path(ingredient)
    exists = check_if_filepath_exists(filepath)
    if not exists:
        print(f"Saving in: {filepath}.")
        information = get_ingredient_info(ingredient)
        save_as_json(what=information, where=filepath)
        time.sleep(0.1)
    else:
        print(f"File of ingredient {ingredient} already exists.")


def get_val_from_dict(x, dictionary):
    """Gets a value from the corresponding dictionary"""
    val = dictionary.get(x)
    if val is None:
        return x
    return val


def get_edamam_ingredient_info_from_flavordb():
    """Gets the ingredients of flavor db, then obtain the corresponding  """
    general_path = get_general_path()
    flavor_db_datapath = join_paths(
        general_path, DATA_FILE_PATH, FLAVORDB_FILE_NAME
    )
    flavordb_df = get_flavor_db_as_df(flavor_db_datapath)
    ingredients_list = flavordb_df.entity_alias_readable.sort_values()
    ingredients_list = ingredients_list[
        ~ingredients_list.isin(not_found_in_edamam)
    ]

    final_ingredient_list = ingredients_list.apply(
        get_val_from_dict, dictionary=found_in_edamam_wierd
    ).to_list()
    for ingredient in final_ingredient_list:
        process_ingredient_data(ingredient)
    return None


def found_in_edamam_wierd_to_lower():
    found_wierd_lower = {}
    for key in found_in_edamam_wierd.keys():
        new_key = key.lower().strip()
        found_wierd_lower[new_key] = found_in_edamam_wierd[key].lower().strip()
    return found_wierd_lower


def changenames(string):
    string = string.lower().strip()
    found_in_edamam_wierd_lower = found_in_edamam_wierd_to_lower()
    if found_in_edamam_wierd_lower.get(string):
        return found_in_edamam_wierd_lower.get(string).replace(' ', '-')
    return string.lower().strip().replace(' ', '-')


def proceed_with_flavor_db_search_names():
    general_path = get_general_path()
    flavor_db_datapath = join_paths(
        general_path, DATA_FILE_PATH, FLAVORDB_FILE_NAME
    )
    flavordb_df = get_flavor_db_as_df(flavor_db_datapath)
    flavordb_df['search_names'] = flavordb_df[
        "entity_alias_readable"
    ].str.lower().apply(changenames)
    return flavordb_df


def get_edamam_list():
    general_path = get_general_path()
    path = join_paths(general_path, DATA_EDAMAM_INGREDIENT_PATH)
    edamam_datalist = os.listdir(path)
    edamam_list = pd.Series(edamam_datalist)\
        .str.replace('edamam_', '')\
        .str.replace('.json', '')
    return edamam_list


def get_json_info(json_path):
    with open(json_path, 'r') as file:
        f = json.load(file)
    return f


def get_similarity(x, ingredient):
    return SequenceMatcher(a=ingredient, b=x).ratio()


def food_ids(ingredient):
    general_path = get_general_path()
    path = join_paths(general_path, DATA_EDAMAM_INGREDIENT_PATH)
    filename = f"edamam_{ingredient}.json"
    filepath = join_paths(path, filename)
    hint_values = pd.DataFrame(
        pd.DataFrame(get_json_info(
            filepath
        )['hints']).food.to_list()
    )
    step1 = hint_values[
        hint_values.label.str.lower().str.replace(' ', '-') == ingredient]
    step2 = list(step1[step1.category == 'Generic foods'].foodId.unique())
    if step2:
        return step2
    most_similar_ingredient = hint_values[
        hint_values.category == 'Generic foods'
        ].label.apply(get_similarity, ingredient=ingredient)
    if len(most_similar_ingredient):
        step3 = hint_values.loc[most_similar_ingredient.argmax()]
        return [step3.foodId]
    most_similar_ingredient2 = hint_values[(
            (hint_values.brand.isnull())
            & (hint_values.category == 'Generic meals')
    )].label.apply(get_similarity, ingredient=ingredient)
    if len(most_similar_ingredient2):
        step4 = hint_values.loc[most_similar_ingredient2.argmax()]
        return [step4.foodId]
    most_similar_ingredient3 = hint_values.label.apply(
        get_similarity, ingredient=ingredient
    )
    if len(most_similar_ingredient3):
        step5 = hint_values.loc[most_similar_ingredient3.argmax()]
        return [step5.foodId]
    return []


def edamam_and_flavor_db():
    general_path = get_general_path()
    path = join_paths(general_path, DATA_EDAMAM_PATH)
    filepath = join_paths(path, EDAMAM_FLAVORDB_FILE_NAME)

    edamam_list = get_edamam_list()
    flavordb_df = proceed_with_flavor_db_search_names()
    edamam_flavor_db = flavordb_df[flavordb_df.search_names.isin(edamam_list)]
    edamam_flavor_db = edamam_flavor_db[USEFULL_COLUMNS]
    edamam_flavor_db['json_path'] = f'{path}edamam_' + \
                                    edamam_flavor_db['search_names'] + \
                                    '.json'
    edamam_flavor_db['possible_food_ids'] = edamam_flavor_db[
        "search_names"
    ].apply(food_ids)
    save_as_pickle(what=edamam_flavor_db, where=filepath)


# Nutritional information
def save_as_edammam_id_food_info(food_id, information):
    general_path = get_general_path()
    path = join_paths(general_path, DATA_EDAMAM_NUTRIENTS_PATH)
    filename = f'edamam__{food_id}.json'
    filepath = join_paths(path, filename)
    with open(filepath, 'w') as f:
        json.dump(information, f)


def check_if_food_info_downloaded(food_id):
    general_path = get_general_path()
    path = join_paths(general_path, DATA_EDAMAM_NUTRIENTS_PATH)
    exists = check_if_filepath_exists(path)
    if not exists:
        print(f'Creating folders: {path}')
        make_desired_folder(path)

    elements_of_path = os.listdir(path)
    all_ids = [
        food_id.replace('edamam__', '').replace('.json', '')
        for food_id
        in elements_of_path
    ]
    if food_id in all_ids:
        filename = f"edamam__{food_id}.json"
        file_path = join_paths(path, filename)
        return file_path
    return None


def get_ingredient_nutritional_info(food_id_list):
    api_id = os.environ["API_ID"]
    api_key = os.environ["API_KEY"]
    url = f'https://api.edamam.com/api/food-database/v2/' \
          f'nutrients?app_id={api_id}&app_key={api_key}'
    save_responses = []

    for food_id in food_id_list:
        headers = {
            "ingredients": [
                {
                    "quantity": 100,
                    "measureURI": "http://www.edamam.com/"
                                  "ontologies/edamam.owl#Measure_gram",
                    "foodId": f"{food_id}"
                }
            ]
        }
        downloaded_path = check_if_food_info_downloaded(food_id)
        if downloaded_path:
            print(f"File already exists. Getting the: {downloaded_path} file.")
            with open(downloaded_path, 'r') as f:
                actual_response = json.load(f)
        else:
            response = requests.post(url, json=headers)
            actual_response = response.json()
            print(f"Downloading information for {food_id}")

            save_as_edammam_id_food_info(food_id, actual_response)

        relevant_nutrient_info = actual_response.get('totalNutrients')
        response_df = pd.DataFrame(relevant_nutrient_info)
        save_responses.append(response_df)
    if save_responses:
        reduced_responses = pd.concat(save_responses)
        nutritional_info = process_nutritional_information_ingredient(
            reduced_responses)
    else:
        nutritional_info = 'NO NUTRITIONAL INFO'
    return nutritional_info


def process_nutritional_information_ingredient(ingredients_df):
    ingredients_df_t = ingredients_df.T
    try:
        ingredients_df_t['real_quantity'] = ingredients_df_t['quantity'].mean(
            axis=1)
    except ValueError:
        ingredients_df_t['real_quantity'] = ingredients_df_t['quantity']
    except KeyError:
        ingredients_df_t['real_quantity'] = None
    ingredients_df_t = ingredients_df_t.loc[
                       :, ~ingredients_df_t.columns.duplicated()
                       ].copy()
    try:
        reduced_ingredients_df_t = ingredients_df_t.drop(columns='quantity')
    except:
        reduced_ingredients_df_t = ingredients_df_t
    return reduced_ingredients_df_t.T.to_dict()


def delete_files(food_id_list):
    for food_id in food_id_list:
        try:
            path = check_if_food_info_downloaded(food_id)
            if path is not None:
                os.remove(path)
        except ValueError as e:
            print(e)


def nutritional_edamam_flavor_db():
    general_path = get_general_path()
    path_edamam = join_paths(general_path, DATA_EDAMAM_PATH)
    path_raw = join_paths(general_path, DATA_FILE_PATH)
    filepath = join_paths(path_edamam, EDAMAM_FLAVORDB_FILE_NAME)
    filepath_updated = join_paths(
        path_raw, EDAMAM_FLAVORDB_NUTRIENTS_FILE_NAME
    )
    load_api_keys()

    edamam_flavor_db = read_pickle_with_pandas(filepath)
    nutritional_info = edamam_flavor_db.possible_food_ids.apply(
        get_ingredient_nutritional_info
    )
    edamam_flavor_db['nutritional_info'] = nutritional_info
    not_missing_edamam_flavors = edamam_flavor_db[
        ~(edamam_flavor_db.nutritional_info == {})
    ]
    save_as_pickle(what=not_missing_edamam_flavors, where=filepath_updated)


if __name__ == "__main__":
    get_edamam_ingredient_info_from_flavordb()
    edamam_and_flavor_db()
    nutritional_edamam_flavor_db()
